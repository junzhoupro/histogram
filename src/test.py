import json
import os

import matplotlib.pyplot as plt
import numpy as np

import PIL.Image
import torch
import torch.nn.functional as F

# source = data.chelsea()
# target = data.coffee()
# source_mask = np.zeros_like(source)
# source_mask[100:200, 100:200] = 1
# target_mask = np.zeros_like(target)
# target_mask[100:200, 100:200] = 1

# matched_np = match_histograms(image=source, reference=target, image_mask=source_mask, reference_mask=target_mask)

# device = torch.device("cuda")
# source_th = torch.from_numpy(source).float().to(device) / 255
# target_th = torch.from_numpy(target).float().to(device) / 255
# source_mask_th = torch.from_numpy(source_mask).float().to(device)
# target_mask_th = torch.from_numpy(target_mask).float().to(device)
# source_th = source_th[None].permute(0, 3, 1, 2)
# target_th = target_th[None].permute(0, 3, 1, 2)
# source_mask_th = source_mask_th[None].permute(0, 3, 1, 2)
# target_mask_th = target_mask_th[None].permute(0, 3, 1, 2)
# print(f"source_th: {source_th.shape}")
# print(f"target_th: {target_th.shape}")
# print(f"source_mask_th: {source_mask_th.shape}")
# print(f"target_mask_th: {target_mask_th.shape}")

# hist_matching = HistogramMatching(num_bins=256, sigma=1e-3)
# target_th.requires_grad_(True)
# matched_th = hist_matching(source_th, target_th, source_mask_th, target_mask_th)
# print(f"matched_th: {matched_th.shape}")
# print(f"{matched_th.requires_grad}")
# matched_th = (matched_th.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255.0).astype(int)
# print(f"matched_th: {matched_th.shape}")
# diff = np.abs(matched_np - matched_th)
# print(f"diff: {diff.max()}")

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(10, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3, ax4):
#     aa.set_axis_off()

# ax1.imshow(source)
# ax1.set_title("Source")
# ax2.imshow(target)
# ax2.set_title("Reference")
# ax3.imshow(matched_np)
# ax3.set_title("Matched (np)")
# ax4.imshow(matched_th)
# ax4.set_title("Matched (pytorch)")

# plt.tight_layout()
# plt.savefig("Figure_1_new.png")
# plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(10, 8))


# for i, img in enumerate((source, target, matched_np, matched_th)):
#     for c, c_color in enumerate(("red", "green", "blue")):
#         img_hist, bins = exposure.histogram(img[..., c])
#         axes[c, i].plot(bins, img_hist / img_hist.max())
#         img_cdf, bins = exposure.cumulative_distribution(img[..., c])
#         axes[c, i].plot(bins, img_cdf)
#         axes[c, 0].set_ylabel(c_color)

# axes[0, 0].set_title("Source")
# axes[0, 1].set_title("Target")
# axes[0, 2].set_title("Matched (np)")
# axes[0, 3].set_title("Matched (pytorch)")

# plt.tight_layout()
# plt.show()

# source = data.chelsea()

# dark = exposure.adjust_gamma(source, gamma=2.2)
# light = exposure.adjust_gamma(source, gamma=0.5)
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3):
#     aa.set_axis_off()

# ax1.imshow(source)
# ax1.set_title("Source")
# ax2.imshow(dark)
# ax2.set_title("Dark (gamma=2.2)")
# ax3.imshow(light)
# ax3.set_title("Light (gamma=0.5)")

# plt.tight_layout()
# plt.show()

# matched_dark = exposure.match_histograms(dark, source, channel_axis=-1)
# matched_light = exposure.match_histograms(light, source, channel_axis=-1)
# print(f"matched_dark: {matched_dark}")
# print(f"matched_light: {matched_light}")
# diff = np.abs(matched_dark - matched_light).clip(0, 255)

# fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=1, ncols=6, figsize=(15, 3), sharex=True, sharey=True)
# for aa in (ax1, ax2, ax3, ax4, ax5, ax6):
#     aa.set_axis_off()

# ax1.imshow(dark)
# ax1.set_title("Dark (gamma=2.2)")
# ax2.imshow(light)
# ax2.set_title("Light (gamma=0.5)")
# ax3.imshow(source)
# ax3.set_title("Target")
# ax4.imshow(matched_dark)
# ax4.set_title("Matched (dark)")
# ax5.imshow(matched_light)
# ax5.set_title("Matched (light)")
# ax6.imshow(diff)
# ax6.set_title("Diff")

# plt.tight_layout()
# plt.show()

# from utils.color import get_rays_np, load_hdr, match_histograms, sample_env_map


# jinx_synthetic_path = "/disk3/proj_viton/jinx-synthetic/blender-files/jinx-synthetic/"
# jewelry_img_name = "037-armring-bangle"
# view_index = 0
# hdr_path = "/disk3/proj_viton/jinx-synthetic/hdrs/"
# hdr_name = "hamburg_hbf_4k"
# cam_transform = f"{jinx_synthetic_path}/transforms.json"

# with open(cam_transform, "r") as fp:
#     meta = json.load(fp)
# H = W = 512
# camera_angle_x = float(meta["camera_angle_x"])
# focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

# image = PIL.Image.open(f"{jinx_synthetic_path}/preprocessed_data/img/{jewelry_img_name}_{view_index:02d}.png")
# image = np.array(image)
# rgb = image[..., :3]
# alpha = image[..., 3:] / 255
# image = (rgb * alpha + (1 - alpha) * 255).astype(np.uint8)
# print(f"image: {image.shape} min: {image.min()} max: {image.max()}")
# mask = PIL.Image.open(f"{jinx_synthetic_path}/preprocessed_data/mask/{jewelry_img_name}_{view_index:02d}.png")
# mask = np.array(mask)[..., :3]
# print(f"mask: {mask.shape} min: {mask.min()} max: {mask.max()}")
# env_hdr = load_hdr(f"{hdr_path}/{hdr_name}.exr")
# env_hdr = torch.tensor(env_hdr, dtype=torch.float32).cuda()

# pose = np.array(meta["view"][view_index]["transform_matrix"]).astype(np.float32)
# rays_o, rays_d = get_rays_np(H, W, focal, pose)
# rays_d = torch.tensor(rays_d, dtype=torch.float32).cuda()
# rays_d = F.normalize(rays_d, dim=-1)  # [H, W, 3]

# sample_hdr = sample_env_map(env_hdr, rays_d)
# bg = sample_hdr.clamp(0, 1)
# bg = bg ** (1 / 2.2)  # gamma correction

# bg = bg.cpu().numpy()
# bg = np.uint8(bg * 255)
# print(f"bg: {bg.shape} min: {bg.min()} max: {bg.max()}")
# # PIL.Image.fromarray(bg).save("bg.png")

# gt = PIL.Image.open(f"{jinx_synthetic_path}/real_data_relighting/{jewelry_img_name}_{view_index:02d}/{hdr_name}.png")
# gt = np.array(gt)
# gt = (gt[..., :3] * alpha + (1 - alpha) * bg).astype(np.uint8)
# composite = (image[..., :3] * alpha + (1 - alpha) * bg).astype(np.uint8)

# fg_mask = mask / 255
# bg_mask = 1 - fg_mask
# matched_bg = match_histograms(image=image, reference=bg, image_mask=fg_mask, reference_mask=bg_mask)
# matched_bg = (matched_bg[..., :3] * alpha + (1 - alpha) * bg).astype(np.uint8)
# matched_gt = match_histograms(image=image, reference=gt, image_mask=fg_mask, reference_mask=fg_mask)
# matched_gt = (matched_gt[..., :3] * alpha + (1 - alpha) * bg).astype(np.uint8)

# fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), sharex=True, sharey=True)
# for axis_y in axes:
#     for axis in axis_y:
#         axis.set_axis_off()

# axes[0, 0].imshow(image)
# axes[0, 0].set_title("Foreground")
# axes[0, 1].imshow(fg_mask)
# axes[0, 1].set_title("Foreground Mask")
# axes[0, 2].imshow(bg)
# axes[0, 2].set_title("Background")
# axes[0, 3].imshow(bg_mask)
# axes[0, 3].set_title("Background Mask")
# axes[1, 0].imshow(composite)
# axes[1, 0].set_title("Composite")
# axes[1, 1].imshow(matched_bg)
# axes[1, 1].set_title("Histogram Matching (bg)")
# axes[1, 2].imshow(matched_gt)
# axes[1, 2].set_title("Histogram Matching (gt)")
# axes[1, 3].imshow(gt)
# axes[1, 3].set_title("Ground Truth")

# plt.tight_layout()
# plt.show()

from models.dataset import JinxSynthetic
from models.hitsogram import HistogramMatching
from skimage import data, exposure, io
from tqdm import tqdm
from utils.color import match_histograms

dataset = JinxSynthetic(img_path="/disk3/proj_viton/jinx-synthetic/blender-files/jinx-synthetic", hdr_path="/disk3/proj_viton/jinx-synthetic/hdrs")
# print(f"len: {len(dataset)}")
# index = np.random.randint(0, len(dataset))
# print(f"index: {index}")
# index = 14883
# sample = dataset[index]
# for k, v in sample.items():
#     print(f"{k}: {v.shape}")
#     v = v.transpose(1, 2, 0)
#     if k == "mask" or k == "alpha":
#         v = v[..., 0]
#     PIL.Image.fromarray((v * 255).astype(np.uint8)).save(f"{k}.png")

# image = (sample["composite"].transpose(1, 2, 0) * 255).astype(np.uint8)
# gt = (sample["gt"].transpose(1, 2, 0) * 255).astype(np.uint8)
# mask = sample["mask"].transpose(1, 2, 0).astype(np.uint8)
# matched_gt = match_histograms(image=image, reference=gt)
# PIL.Image.fromarray(matched_gt).save("test_1.png")

# histogram_matching = HistogramMatching(num_bins=256, sigma=0.001)
# bins = torch.linspace(0, 1, histogram_matching.num_bins).cuda()
# hist = []
# for sample in tqdm(dataset):
#     gt = torch.tensor(sample["gt"], dtype=torch.float32).cuda()
#     alpha = torch.tensor(sample["alpha"], dtype=torch.float32).cuda()
#     hist.append(histogram_matching.soft_histogram(gt[None], bins=bins, mask=alpha).cpu().numpy())
# hist = np.concatenate(hist)
# print(f"hist: {hist.shape}")
# hist_mean = hist.mean(axis=0)
# hist_mean.astype(np.float32).tofile(f"assets/hist_mean.bin")
# hist_var = hist.var()
# np.savetxt(f"assets/hist_var.txt", [hist_var])

hist = []
for fname in tqdm(dataset.all_fnames):
    image = io.imread(fname)
    rgb, alpha = image[..., :3], image[..., 3:]
    mask = np.rint(alpha).astype(bool)
    mask = np.repeat(mask, repeats=3, axis=-1)
    # print(f"rgb: {rgb.shape} mask: {mask.shape}")
    rgb = np.ma.array(rgb, mask=~mask)
    hist_r, bins = exposure.histogram(rgb[..., 0].compressed(), source_range="dtype", normalize=True)
    hist_g, bins = exposure.histogram(rgb[..., 1].compressed(), source_range="dtype", normalize=True)
    hist_b, bins = exposure.histogram(rgb[..., 2].compressed(), source_range="dtype", normalize=True)
    hist.append(np.stack([hist_r, hist_g, hist_b]))
hist = np.stack(hist)
print(f"hist: {hist.shape}")
hist_mean = hist.mean(axis=0)
hist_mean.astype(np.float32).tofile(f"assets/hist_mean.bin")
hist_var = hist.var()
np.savetxt(f"assets/hist_var.txt", [hist_var])
