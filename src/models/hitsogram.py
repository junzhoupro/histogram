from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramMatching(nn.Module):
    def __init__(self, num_bins: int = 256, sigma: float = 0.01):
        """
        Differentiable histogram matching module.
        Args:
            num_bins: Number of bins for the histogram (default: 256)
            sigma: Bandwidth for the Gaussian kernel in soft histogram (default: 0.01)
        """
        super().__init__()
        self.num_bins = num_bins
        self.sigma = sigma

    def forward(self, source: torch.Tensor, target: torch.Tensor, source_mask: Optional[torch.Tensor] = None, target_mask: Optional[torch.Tensor] = None):
        """
        Args:
            source: Tensor of shape (N, C, H, W), values in [0, 1]
            target: Tensor of shape (N, C, H, W), values in [0, 1], or Tensor of shape (N, C, B), which is the target histogram directly.
            source_mask: (N, C, H, W) or (N, 1, H, W) or None
            target_mask: (N, C, H, W) or (N, 1, H, W) or None
        Returns:
            matched: Tensor of shape (N, C, H, W) with matched histograms
        """
        device = source.device
        N, C, H, W = source.shape
        bins = torch.linspace(0, 1, self.num_bins, device=device)

        # Estimate soft histograms
        if source_mask is None:
            source_mask = torch.ones_like(source)
        if source_mask.shape[1] == 1:
            source_mask = source_mask.expand(-1, C, -1, -1)
        hist_src = self.soft_histogram(source, bins, mask=source_mask)  # (N, C, B)

        if target.ndim == 4:
            if target_mask is None:
                target_mask = torch.ones_like(target)
            if target_mask.shape[1] == 1:
                target_mask = target_mask.expand(-1, C, -1, -1)
            hist_tgt = self.soft_histogram(target, bins, mask=target_mask)  # (N, C, B)
        elif target.ndim == 3:
            assert target.shape[1] == C and target.shape[2] == self.num_bins, "Target histogram is in wrong shape"
            hist_tgt = target
        else:
            raise NotImplementedError(f"Unsupported target tensor of shape {target.shape}")

        # Estimate CDFs
        cdf_src = torch.cumsum(hist_src, dim=-1)
        cdf_tgt = torch.cumsum(hist_tgt, dim=-1)

        # Flatten source image
        source_flat = source.flatten(start_dim=2)  # (N, C, H x W)
        mask_flat = source_mask.flatten(start_dim=2)  # (N, C, H x W)
        matched_flat = torch.clone(source_flat)

        for b in range(N):
            for c in range(C):
                x = source_flat[b, c]  # (H x W,)
                m = mask_flat[b, c]

                if m.sum() < 1e-4:
                    continue  # Skip empty mask

                src_cdf_vals = F.interpolate(cdf_src[b, c][None, None], size=self.num_bins, mode="linear", align_corners=True)[0, 0]
                src_cdf_vals = torch.clamp(src_cdf_vals, 0, 1)

                x_bin_indices = (x * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)
                cdf_values = src_cdf_vals[x_bin_indices]

                tgt_cdf_vals = cdf_tgt[b, c]
                bin_vals = bins

                # Linear inverse mapping via interpolation
                x_mapped = self._interp1d(cdf_values, tgt_cdf_vals, bin_vals)
                # Replace only in the masked region
                matched_flat[b, c] = x * (1 - m) + x_mapped * m

        return matched_flat.reshape(N, C, H, W).clamp(min=0, max=1)

    def soft_histogram(self, x: torch.Tensor, bins: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        N, C, H, W = x.shape
        x = x.reshape(N, C, -1, 1)  # (N, C, H x W, 1)
        bins = bins.view(1, 1, 1, -1)  # (1, 1, 1, B)
        weights = torch.exp(-0.5 * ((x - bins) / self.sigma) ** 2)
        if mask is not None:
            if mask.shape[1] == C:
                mask = mask.reshape(N, C, -1, 1)
            else:
                mask = mask.reshape(N, 1, -1, 1).expand(-1, C, -1, -1)
            weights = weights * mask

        hist = weights.sum(dim=2)  # (N, C, B)
        # print(f"hist: {hist}")
        return hist / (hist.sum(dim=-1, keepdim=True) + 1e-6)

    def _interp1d(self, x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
        """
        Linearly interpolate x given xp (monotonic increasing) and fp
        Args:
            x: (...,) values in [0, 1] to be mapped
            xp: (B,) source x-axis (e.g., CDF values)
            fp: (B,) target y-axis (e.g., bin centers)
        Returns:
            (...,) interpolated values
        """
        # Flatten for batched ops
        original_shape = x.shape
        x = x.reshape(-1)

        # Find bin indices
        idx = torch.searchsorted(xp, x, right=True).clamp(1, len(xp) - 1)
        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        # Linear interpolate
        t = (x - x0) / (x1 - x0 + 1e-6)
        y = y0 + t * (y1 - y0)
        return y.view(original_shape)
