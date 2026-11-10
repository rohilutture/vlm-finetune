"""Helpers for the flattened vision inputs of Qwen2-VL / Qwen2.5-VL.

pixel_values is the patches of ALL images in the batch concatenated along dim 0, and image_grid_thw [N, 3]
gives (t, h, w) patch counts per image (one image per sample here), so image i owns rows
[offset_i, offset_i + t*h*w). Slicing a sub-batch therefore needs the matching row ranges.
"""
from __future__ import annotations

import torch


def slice_visual(pixel_values: torch.Tensor, image_grid_thw: torch.Tensor, idx: list[int] | torch.Tensor):
    """Visual inputs for samples `idx` (one image per sample, in batch order)."""
    idx = [int(i) for i in idx]
    counts = image_grid_thw.prod(dim=1).tolist()
    offsets = [0]
    for c in counts:
        offsets.append(offsets[-1] + int(c))
    pv = torch.cat([pixel_values[offsets[i]:offsets[i + 1]] for i in idx], dim=0)
    return pv, image_grid_thw[idx]
