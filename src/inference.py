import glob
import logging
import os
import random
from argparse import ArgumentParser

import numpy as np
import PIL.Image
import torch
import torchvision

from models import HistogramMatching, JinxSynthetic, Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.visualize import plot_rgb_histogram, plot_single_rgb_histogram


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

    # Create logger:
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    # Create model:
    logger.info(f"[Inference] Setting up Model from {args.checkpoint}...")
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

    jewelry_list = os.listdir(f"{args.img_dir}/render_default")
    hdr_list = os.listdir(f"{args.img_dir}/relit")
    os.makedirs(args.outdir, exist_ok=True)
    for jewelry_name in tqdm(jewelry_list):
        default = PIL.Image.open(f"{args.img_dir}/render_default/{jewelry_name}/{jewelry_name}.png")
        default = np.array(default)[..., :3] / 255  # [0, 255] => [0, 1]
        default = default.transpose(2, 0, 1)  # (3, H, W)
        default = torch.tensor(default[None], dtype=torch.float32, device=device)
        for hdr_name in hdr_list:
            os.makedirs(f"{args.outdir}/{hdr_name}", exist_ok=True)
            relit = PIL.Image.open(f"{args.img_dir}/relit/{hdr_name}/{jewelry_name}0001.png")
            mask = PIL.Image.open(f"{args.img_dir}/relit/{hdr_name}/{jewelry_name}_mask0001.png")
            relit = np.array(relit)[..., :3] / 255  # [0, 255] => [0, 1]
            relit = relit.transpose(2, 0, 1)  # (3, H, W)
            mask = np.array(mask)[..., :1] / 255  # [0, 255] => [0, 1]
            mask = mask.transpose(2, 0, 1)  # (1, H, W)

            relit = torch.tensor(relit[None], dtype=torch.float32, device=device)
            mask = torch.tensor(mask[None], dtype=torch.float32, device=device)
            composite = default * mask + relit * (1 - mask)  # (1, 3, H, W)

            fg_hist = histogram_matching.soft_histogram(default, bins=bins, mask=mask)  # (1, 3, B)
            bg_hist = histogram_matching.soft_histogram(relit, bins=bins, mask=1 - mask)  # (1, 3, B)
            x = (fg_hist - hist_mean[None]) / hist_std  # Normalize foreground histogram as input
            c = (bg_hist - hist_mean[None]) / hist_std  # Normalize background histogram as condition
            pred_hist = model(x=x, c=c)  # (1, 3, B)
            image = histogram_matching(source=composite, target=pred_hist, source_mask=mask)  # (1, 3, H, W)

            torchvision.utils.save_image(image, f"{args.outdir}/{hdr_name}/{jewelry_name}_hist.png")
            torchvision.utils.save_image(composite, f"{args.outdir}/{hdr_name}/{jewelry_name}_composite.png")
            torchvision.utils.save_image(relit, f"{args.outdir}/{hdr_name}/{jewelry_name}_gt.png")

    logger.info("[Inference] Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required.
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
