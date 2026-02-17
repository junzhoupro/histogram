import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from models import HistogramMatching, JinxSynthetic, Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualize import plot_rgb_histogram


#################################################################################
#                                  Evaluation Loop                              #
#################################################################################


def main(args):
    # Setup Pytorch:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.outdir, exist_ok=True)

    # Create logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Setup data:
    logger.info("[Eval] Setting up Dataset...")
    dataset = JinxSynthetic(img_path=args.img_path, hdr_path=args.hdr_path)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"[Eval] Dataset contains {len(dataset):,} images ({dataset.num_jewelries} jewelries, {dataset.num_views} views, {dataset.num_hdrs} HDRs)")

    # Create model:
    logger.info(f"[Eval] Setting up Model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model_args = checkpoint["args"]
    model = Transformer(
        in_channels=3,
        n_heads=model_args.n_heads,
        d_head=model_args.d_head,
        mlp_ratio=model_args.mlp_ratio,
        depth=model_args.depth,
        dropout=model_args.dropout,
        out_activation=model_args.out_activation,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    histogram_matching = HistogramMatching(num_bins=256, sigma=0.001).to(device)

    # Histogram assets:
    hist_mean = np.fromfile("assets/hist_mean.bin", dtype=np.float32).reshape(3, -1)
    hist_std = float(np.genfromtxt("assets/hist_var.txt") ** 0.5)
    hist_mean = torch.tensor(hist_mean, dtype=torch.float32, device=device)
    bins = torch.linspace(0, 1, histogram_matching.num_bins, device=device)

    logger.info(f"[Eval] Beginning evaluation...")
    for idx, batch in enumerate(tqdm(data_loader)):
        if idx % (dataset.num_views * dataset.num_hdrs // args.batch_size) != 0:
            continue

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

        torchvision.utils.save_image(composite, f"{args.outdir}/composite_{idx:04d}.png")
        torchvision.utils.save_image(gt, f"{args.outdir}/gt_{idx:04d}.png")
        torchvision.utils.save_image(image, f"{args.outdir}/pred_{idx:04d}.png")
        plot_rgb_histogram(fp=f"{args.outdir}/hist_{idx:04d}.png", fg_hist=fg_hist.cpu().numpy(), pred_hist=pred_hist.cpu().numpy(), gt_hist=gt_hist.cpu().numpy())

    logger.info("[Eval] Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--hdr_path", type=str, required=True)
    # Eval hyperparameters.
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
