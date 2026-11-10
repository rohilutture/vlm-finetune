"""Test doubles: a character-level fake tokenizer/processor with the real Qwen2.5-VL processor's interface and a tiny
random-weight Qwen2.5-VL. The model is the REAL transformers class (so LoRA targeting, rope index with left padding,
generate() with pixel_values, etc. are exercised for real); only the tokenizer/image preprocessing is faked."""
from __future__ import annotations

import numpy as np
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

PAD, EOS = 0, 1
IMG, VSTART, VEND = 250, 252, 253
CHAR0 = 3  # 7-bit ASCII (including newline) -> ids 3..130


class Batch(dict):
    def to(self, device):
        return Batch({k: (v.to(device) if hasattr(v, "to") else v) for k, v in self.items()})


class FakeTokenizer:
    padding_side = "right"
    pad_token_id = PAD
    eos_token_id = EOS

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            if s.startswith("<|im_end|>", i):
                out.append(EOS); i += len("<|im_end|>")
            elif s.startswith("<image>", i):
                out.append(-1); i += len("<image>")  # placeholder, expanded by the processor
            else:
                assert ord(s[i]) < 128, f"fake tokenizer is ASCII only: {s[i]!r}"
                out.append(ord(s[i]) + CHAR0); i += 1
        return out

    def convert_tokens_to_ids(self, tok: str) -> int:
        return EOS if tok == "<|im_end|>" else -1


class FakeProcessor:
    """Image -> 28x28 -> grid (1,2,2) -> 4 patches -> 1 visual token after the 2x2 merge."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(c["text"] for c in messages[0]["content"] if c["type"] == "text")
        return "U:<image>" + text + "\nA:"

    def __call__(self, text, images, return_tensors="pt", padding=True):
        seqs, pixels, grids = [], [], []
        for t, im in zip(text, images):
            a = np.asarray(im.convert("RGB").resize((28, 28))).astype(np.float32) / 127.5 - 1.0  # [28,28,3]
            patches = a.reshape(2, 14, 2, 14, 3).transpose(0, 2, 1, 3, 4).reshape(4, 14, 14, 3)  # 4 patches
            pix = np.repeat(patches.transpose(0, 3, 1, 2)[:, :, None], 2, axis=2).reshape(4, -1)  # [4, 3*2*14*14]
            pixels.append(pix)
            grids.append([1, 2, 2])
            ids = []
            for tok in self.tokenizer.encode(t):
                ids += [VSTART, IMG, VEND] if tok == -1 else [tok]
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        left = self.tokenizer.padding_side == "left"
        ids, mask = [], []
        for s in seqs:
            pad = [PAD] * (width - len(s))
            ids.append(pad + s if left else s + pad)
            mask.append([0] * len(pad) + [1] * len(s) if left else [1] * len(s) + [0] * len(pad))
        return Batch(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask),
                     pixel_values=torch.tensor(np.concatenate(pixels), dtype=torch.float32),
                     image_grid_thw=torch.tensor(grids))

    def batch_decode(self, ids, skip_special_tokens=True):
        out = []
        for row in ids.tolist():
            out.append("".join(chr(i - CHAR0) for i in row if CHAR0 <= i < CHAR0 + 128))
        return out


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Qwen2_5_VLConfig(
        text_config=dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=2048,
                         rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
                         image_token_id=IMG, video_token_id=251, vision_start_token_id=VSTART, vision_end_token_id=VEND,
                         bos_token_id=2, eos_token_id=EOS, pad_token_id=PAD),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=14,
                           spatial_merge_size=2, temporal_patch_size=2, in_channels=3, fullatt_block_indexes=[1], window_size=56),
        image_token_id=IMG, video_token_id=251, vision_start_token_id=VSTART, vision_end_token_id=VEND,
    )
    m = Qwen2_5_VLForConditionalGeneration(cfg).float()
    m.generation_config.eos_token_id = [EOS]
    m.generation_config.pad_token_id = PAD
    return m
