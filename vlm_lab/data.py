"""Chat formatting, label masking and batch collation for Qwen2.5-VL style processors.

The processor interface used (so tests can substitute a fake one):
    processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=bool) -> str
    processor(text=[str], images=[PIL], return_tensors="pt", padding=True) -> dict with input_ids, attention_mask,
                                                                              pixel_values, image_grid_thw
    processor.tokenizer.padding_side / pad_token_id / eos_token_id, processor.batch_decode(ids, skip_special_tokens=True)
"""
from __future__ import annotations

import torch

from .tasks import Example

IM_END = "<|im_end|>"  # end of an assistant turn in the Qwen chat template


def build_messages(ex: Example) -> list[dict]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ex.prompt()}]}]


def mask_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: torch.Tensor) -> torch.Tensor:
    """Supervise only the assistant's response: -100 on padding and on the first prompt_len tokens of each
    row (the image tokens and the question). Assumes RIGHT padding, so each row starts at index 0."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for i, n in enumerate(prompt_lens.tolist()):
        labels[i, :n] = -100
    return labels


class _Padding:
    """Context manager: temporarily set the tokenizer's padding side."""

    def __init__(self, processor, side):
        self.tok, self.side = processor.tokenizer, side

    def __enter__(self):
        self.old = self.tok.padding_side
        self.tok.padding_side = self.side

    def __exit__(self, *a):
        self.tok.padding_side = self.old


class SFTCollator:
    """Batches (image, question, response) into model inputs with labels on the response only."""

    def __init__(self, processor, end_token: str = IM_END):
        self.p, self.end = processor, end_token

    def __call__(self, examples: list[Example]) -> dict:
        images = [e.image() for e in examples]
        prompts = [self.p.apply_chat_template(build_messages(e), tokenize=False, add_generation_prompt=True) for e in examples]
        fulls = [pr + e.response() + self.end for pr, e in zip(prompts, examples)]
        with _Padding(self.p, "right"):
            prompt_in = self.p(text=prompts, images=images, return_tensors="pt", padding=True)
            full = self.p(text=fulls, images=images, return_tensors="pt", padding=True)
        prompt_lens = prompt_in["attention_mask"].sum(dim=1)
        full["labels"] = mask_labels(full["input_ids"], full["attention_mask"], prompt_lens)
        return full


class GenCollator:
    """Prompts only, LEFT padded, for generation. `prompt_len` is the padded prompt length: generated tokens
    start at that index."""

    def __init__(self, processor):
        self.p = processor

    def __call__(self, examples: list[Example]) -> dict:
        images = [e.image() for e in examples]
        prompts = [self.p.apply_chat_template(build_messages(e), tokenize=False, add_generation_prompt=True) for e in examples]
        with _Padding(self.p, "left"):
            return self.p(text=prompts, images=images, return_tensors="pt", padding=True)


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
