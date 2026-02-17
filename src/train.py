import logging
import os
import random
from argparse import ArgumentParser, Namespace
from glob import glob
from time import time

import lpips

import numpy as np
import torch
import torch.distributed as dist
import torchvision

from models import HistogramMatching, JinxSynthetic, Transformer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO, format="[\033[34m%(asctime)s\033[0m] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def kl_divergence(p, q, dim, eps=1e-6):
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    # Normalize to ensure proper distributions
    p = p / p.sum(dim=dim, keepdim=True)
    q = q / q.sum(dim=dim, keepdim=True)

    kl = p * (p.log() - q.log())
    return kl.sum(dim=dim).mean()


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # torch.autograd.set_detect_anomaly(True)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    num_gpus = dist.get_world_size()
    seed = args.global_seed * num_gpus + rank
    global_batch_size = num_gpus * args.batch_size
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={num_gpus}.")

    # Setup an experiment folder:
    if rank == 0:
        # Make results folder (holds all experiment subfolders)
        os.makedirs(args.outdir, exist_ok=True)
        experiment_index = len(glob(f"{args.outdir}/*"))
        # Create an experiment folder
        experiment_dir = f"{args.outdir}/{experiment_index:05d}-gpus{num_gpus:d}-batch{global_batch_size:d}"
        if args.desc != None:
            experiment_dir = f"{experiment_dir}-{args.desc}"
        # Stores saved model checkpoints
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        with open(os.path.join(experiment_dir, "cfg_args"), "w") as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))
        try:
            import torch.utils.tensorboard as tensorboard

            stats_tfevents = tensorboard.SummaryWriter(experiment_dir)
        except ImportError as err:
            logger.error(f"Skipping tfevents export: {err}")
    else:
        logger = create_logger(None)
        stats_tfevents = None

    # Setup data:
    logger.info("[Train] Setting up Dataset...")
    dataset = JinxSynthetic(img_path=args.img_path, hdr_path=args.hdr_path)
    sampler = DistributedSampler(
        dataset,
        num_replicas=num_gpus,
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"[Train] Dataset contains {len(dataset):,} images ({dataset.num_jewelries} jewelries, {dataset.num_views} views, {dataset.num_hdrs} HDRs)")

    # Create model:
    logger.info("[Train] Setting up Model...")
    model = Transformer(
        in_channels=1 if args.separated_channels else 3,
        n_heads=args.n_heads,
        d_head=args.d_head,
        mlp_ratio=args.mlp_ratio,
        depth=args.depth,
        dropout=args.dropout,
        out_activation=args.out_activation,
        separated_channels=args.separated_channels,
    )
    model = DDP(model.to(device), device_ids=[rank])
    histogram_matching = HistogramMatching(num_bins=256, sigma=0.001).to(device)
    logger.info(f"[Train] Transformer Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Histogram assets:
    hist_mean = np.fromfile("assets/hist_mean.bin", dtype=np.float32).reshape(3, -1)
    hist_std = float(np.genfromtxt("assets/hist_var.txt") ** 0.5)
    hist_mean = torch.tensor(hist_mean, dtype=torch.float32, device=device)
    bins = torch.linspace(0, 1, histogram_matching.num_bins, device=device)

    # Setup optimizer:
    logger.info("[Train] Setting up Optimizer...")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lpips_loss = lpips.LPIPS(net="vgg").to(device)

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout

    # Variables for monitoring/logging purposes:
    train_step = 0
    log_step = 0
    running_loss = 0
    running_image = 0
    running_lpips = 0
    running_hist = 0
    start_time = time()

    logger.info(f"[Train] Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"[Train] Beginning epoch {epoch}...")
        for batch in data_loader:
            # Copy batch to GPU.
            composite = batch["composite"].to(device).float()  # (N, 3, H, W)
            gt = batch["gt"].to(device).float()  # (N, 3, H, W)
            bg = batch["bg"].to(device).float()  # (N, 3, H, W)
            mask = batch["mask"].to(device).float()  # (N, 1, H, W)
            alpha = batch["alpha"].to(device).float()  # (N, 1, H, W)

            fg_hist = histogram_matching.soft_histogram(composite, bins=bins, mask=alpha)  # (N, 3, B)
            bg_hist = histogram_matching.soft_histogram(composite, bins=bins, mask=1 - alpha)  # (N, 3, B)
            gt_hist = histogram_matching.soft_histogram(gt, bins=bins, mask=alpha)  # (N, 3, B)

            # Network inference.
            x = (fg_hist - hist_mean[None]) / hist_std  # Normalize foreground histogram as input
            c = (bg_hist - hist_mean[None]) / hist_std  # Normalize background histogram as condition
            pred_hist = model(x=x, c=c)  # (N, 3, B)
            image = histogram_matching(source=composite, target=pred_hist, source_mask=alpha)  # (N, 3, H, W)
            # image = image * alpha + bg * (1 - alpha)  # (N, 3, H, W)
            # print(f"pred_hist: {pred_hist.requires_grad}")
            # print(f"image: {image.requires_grad}")

            # test = histogram_matching(source=composite, target=gt_hist, source_mask=alpha)
            # print(f"test: {test.shape} min: {test.min()} max: {test.max()}")
            # torchvision.utils.save_image(composite, "composite.png")
            # torchvision.utils.save_image(gt, "gt.png")
            # torchvision.utils.save_image(bg, "bg.png")
            # torchvision.utils.save_image(alpha, "alpha.png")
            # torchvision.utils.save_image(image, "pred.png")
            # torchvision.utils.save_image(test, "test.png")
            # exit()

            # Evaluate losses.
            if args.lambda_image > 0:
                image_recon = torch.mean(torch.abs(image - gt))
            else:
                image_recon = torch.zeros([1], dtype=torch.float32, device=device)
            if args.lambda_lpips > 0:
                perceptual = lpips_loss(image * 2 - 1, gt * 2 - 1).mean()
            else:
                perceptual = torch.zeros([1], dtype=torch.float32, device=device)
            if args.lambda_hist > 0:
                if args.hist_loss == "kl":
                    hist_recon = kl_divergence(p=gt_hist, q=pred_hist, dim=-1)
                else:
                    hist_recon = torch.mean(torch.abs(gt_hist - pred_hist))
            else:
                hist_recon = torch.zeros([1], dtype=torch.float32, device=device)
            loss = args.lambda_image * image_recon + args.lambda_lpips * perceptual + args.lambda_hist * hist_recon

            # Compute gradients and update.
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Log loss values:
            running_loss += loss.item()
            running_image += image_recon.item()
            running_lpips += perceptual.item()
            running_hist += hist_recon.item()
            train_step += 1
            log_step += 1
            if train_step % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_step / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_step, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / num_gpus
                avg_image = torch.tensor(running_image / log_step, device=device)
                dist.all_reduce(avg_image, op=dist.ReduceOp.SUM)
                avg_image = avg_image.item() / num_gpus
                avg_lpips = torch.tensor(running_lpips / log_step, device=device)
                dist.all_reduce(avg_lpips, op=dist.ReduceOp.SUM)
                avg_lpips = avg_lpips.item() / num_gpus
                avg_hist = torch.tensor(running_hist / log_step, device=device)
                dist.all_reduce(avg_hist, op=dist.ReduceOp.SUM)
                avg_hist = avg_hist.item() / num_gpus
                logger.info(
                    f"[Train] (step={train_step:07d}) Train Loss: {avg_loss:.4f}, Image Loss: {avg_image:.4f}, LPIPS Loss: {avg_lpips:.4f}, Histogram Loss: {avg_hist:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if stats_tfevents is not None:
                    stats_tfevents.add_scalar("Loss/total", avg_loss, global_step=train_step)
                    stats_tfevents.add_scalar("Loss/image", avg_image, global_step=train_step)
                    stats_tfevents.add_scalar("Loss/lpips", avg_lpips, global_step=train_step)
                    stats_tfevents.add_scalar("Loss/hist", avg_hist, global_step=train_step)
                # Reset monitoring variables:
                running_loss = 0
                running_hist = 0
                running_image = 0
                running_lpips = 0
                log_step = 0
                start_time = time()

        # Save checkpoint:
        if epoch % args.ckpt_every == 0 or epoch == args.epochs - 1:
            if rank == 0:
                checkpoint = {
                    "model": model.module.state_dict(),
                    "args": args,
                    "step": train_step,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch{epoch:05d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"[Train] Saved checkpoint to {checkpoint_path}")
            dist.barrier()

    logger.info("[Train] Done!")
    cleanup()


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required.
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--hdr_path", type=str, required=True)
    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=100)
    # Network architecture settings.
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_head", type=int, default=64)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--out_activation", type=str, choices=["relu", "softmax", "log_softmax"], default="relu")
    parser.add_argument("--separated_channels", action="store_true")
    # Loss hyperparameters.
    parser.add_argument("--lambda_image", type=float, default=1)
    parser.add_argument("--lambda_lpips", type=float, default=1)
    parser.add_argument("--hist_loss", type=str, choices=["kl", "l1"], default="kl")
    parser.add_argument("--lambda_hist", type=float, default=1)
    # Misc.
    parser.add_argument("--desc", type=str, default=None)
    args = parser.parse_args()
    main(args)
