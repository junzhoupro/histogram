import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage import exposure

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def load_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    try:
        with open(path, "rb") as h:
            buffer_ = np.frombuffer(h.read(), np.uint8)
        bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading HDR file {path}: {e}")
        return None
    return rgb


def get_rays_np(H: int, W: int, focal: float, c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
    dirs = np.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d


def sample_env_map(env_map: torch.Tensor, rays: torch.Tensor) -> torch.Tensor:
    env_map = env_map.permute(2, 0, 1).unsqueeze(0)
    theta = torch.atan2(rays[..., 1], rays[..., 0])
    phi = torch.arccos(rays[..., 2])
    # normalize to [-1, 1]
    u = -theta / np.pi
    v = phi / np.pi * 2 - 1
    grid = torch.stack([u, v], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    env_samples = F.grid_sample(env_map, grid, align_corners=True).squeeze().permute(1, 2, 0)  # [H, W, 3]

    return env_samples


def match_histograms(image: np.ndarray, reference: np.ndarray, image_mask: Optional[np.ndarray] = None, reference_mask: Optional[np.ndarray] = None, fill_value: float = 0) -> np.ndarray:
    """Histogram matching with masked image, adapted from https://gist.github.com/tayden/dcc83424ce55bfb970f60db3d4ddad18.
    Args:
        image: np.ndarray, input image (H, W, C)
        reference: np.ndarray, reference image (same shape as source)
        image_mask: binary np.ndarray (H, W, 1), 1 for selected regions to modify, 0 for selected regions to retain.
        reference_mask: binary np.ndarray (H, W, 1), 1 for selected regions to modify, 0 for selected regions to retain.

    Returns:
        matched: np.ndarray, same shape as source, with only masked area modified
    """
    if image_mask is None:
        image_mask = np.ones_like(image)
    image_mask = image_mask.astype(bool)
    if image_mask.shape[-1] == 1:
        image_mask = np.repeat(image_mask, repeats=image.shape[-1], axis=-1)
    if reference_mask is None:
        reference_mask = np.ones_like(reference)
    reference_mask = reference_mask.astype(bool)
    if reference_mask.shape[-1] == 1:
        reference_mask = np.repeat(reference_mask, repeats=reference.shape[-1], axis=-1)

    masked_source_image = np.ma.array(image, mask=~image_mask)
    masked_reference_image = np.ma.array(reference, mask=~reference_mask)

    matched = image.copy()

    for channel in range(masked_source_image.shape[-1]):
        matched_channel = exposure.match_histograms(masked_source_image[..., channel].compressed(), masked_reference_image[..., channel].compressed())

        # Re-insert masked background
        mask_ch = image_mask[..., channel]
        matched[..., channel][mask_ch] = matched_channel.ravel()

    return matched
