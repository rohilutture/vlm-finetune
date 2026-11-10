"""Loading the model and processor. Nothing here downloads unless you pass a hub id and have network access;
pass a local directory to stay offline."""
from __future__ import annotations

import torch


def load_processor(path: str, min_pixels: int = 56 * 56, max_pixels: int = 336 * 336):
    """max_pixels bounds the visual tokens per image: 224x224 -> 64 tokens, 336x336 -> 144 (patch 14, 2x2 merge)."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path, min_pixels=min_pixels, max_pixels=max_pixels)


def load_model(path: str, dtype=torch.bfloat16, device: str = "cuda"):
    from transformers import Qwen2_5_VLForConditionalGeneration

    m = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    return m.to(device)
