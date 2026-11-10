"""GRPO on the procedural visual tasks with an exact verifier as the reward (no reward model).

    python -m vlm_lab.grpo --config configs/grpo.yaml

Per step: take P prompts, repeat each G times (so a group = G samples of the same image+question), sample
completions, score them with verify.total_reward, compute group-relative advantages, then take one clipped-surrogate
update with a k3 KL penalty to a frozen reference. The reference is the base model with the LoRA adapter disabled, or,
when starting from an SFT adapter, a second frozen copy of that adapter.

Alignment: prompts are LEFT padded, so every generated completion starts at index L = padded prompt length, and the
policy/reference log-probs are read from logits[:, L-1:-1] over the same [prompt | completion] layout.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import yaml

from .data import GenCollator, to_device
from .grpo_math import group_advantages, grpo_loss
from .tasks import TASKS, make_split
from .verify import accuracy_reward, format_reward, total_reward
from .vision_utils import slice_visual


def completion_mask(tail: torch.Tensor, stop_ids: set[int]) -> torch.Tensor:
    """1 for tokens up to and including the first stop token, 0 after it (padding)."""
    is_stop = torch.zeros_like(tail, dtype=torch.bool)
    for s in stop_ids:
        is_stop |= tail == s
    after = (is_stop.cumsum(1) - is_stop.long()) > 0
    return (~after).long()


def completion_logprobs(model, full_ids, full_mask, pixel_values, grid_thw, L: int) -> torch.Tensor:
    """Log-probs of the tokens at positions L.. of full_ids given everything before them. Shape [B, T]."""
    out = model(input_ids=full_ids, attention_mask=full_mask, pixel_values=pixel_values, image_grid_thw=grid_thw)
    logits = out.logits[:, L - 1 : -1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, full_ids[:, L:].unsqueeze(-1)).squeeze(-1)


@contextmanager
def reference_policy(model, has_ref_adapter: bool):
    if has_ref_adapter:
        model.set_adapter("ref")
        try:
            yield
        finally:
            model.set_adapter("default")
    else:
        with model.disable_adapter():
            yield


def default_stop_ids(processor, model) -> set[int]:
    tok = processor.tokenizer
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    try:
        ids.add(int(tok.convert_tokens_to_ids("<|im_end|>")))
    except Exception:  # noqa: BLE001
        pass
    ge = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if ge is not None:
        ids.update(ge if isinstance(ge, (list, tuple)) else [ge])
    return {i for i in ids if i is not None and i >= 0}


def grpo_train(model, processor, examples, cfg: dict, device, has_ref_adapter: bool = False, eval_fn=None,
               log=print) -> list[dict]:
    P, G, T = cfg["prompts_per_step"], cfg["group_size"], cfg["max_new_tokens"]
    stop_ids = set(cfg["stop_ids"]) if cfg.get("stop_ids") else default_stop_ids(processor, model)
    # The policy must never SAMPLE the image/video placeholder tokens: the teacher-forced forward pass would then
    # see placeholders with no matching image features and raise. (Near-zero probability for a real model, but not
    # zero, and a crash hours into an RL run is expensive.)
    mcfg = getattr(model, "config", None)
    suppress = sorted({int(getattr(mcfg, k)) for k in ("image_token_id", "video_token_id", "vision_start_token_id",
                                                       "vision_end_token_id") if getattr(mcfg, k, None) is not None})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    gen_collate = GenCollator(processor)
    tok = processor.tokenizer
    rng = random.Random(cfg.get("seed", 0))
    order = list(range(len(examples)))
    rng.shuffle(order)
    out = Path(cfg.get("out_dir", "runs/grpo"))
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.jsonl").open("a")
    history, t0 = [], time.time()

    for step in range(cfg["steps"]):
        picks = [examples[order[(step * P + i) % len(order)]] for i in range(P)]
        reps = [e for e in picks for _ in range(G)]
        inputs = to_device(gen_collate(reps), device)
        L = inputs["input_ids"].shape[1]

        # ---- rollout
        model.eval()
        with torch.no_grad():
            gen = model.generate(**inputs, do_sample=True, temperature=cfg.get("temperature", 1.0), top_p=1.0,
                                 max_new_tokens=T, pad_token_id=tok.pad_token_id, use_cache=True,
                                 suppress_tokens=suppress or None)
        comp = gen[:, L:]
        cmask = completion_mask(comp, stop_ids)
        texts = processor.batch_decode(comp, skip_special_tokens=True)

        # ---- rewards
        rewards = torch.tensor([total_reward(t, e.answer) for t, e in zip(texts, reps)], dtype=torch.float32).view(P, G)
        acc = sum(accuracy_reward(t, e.answer) for t, e in zip(texts, reps)) / len(reps)
        fmt = sum(format_reward(t) for t in texts) / len(texts)
        adv = group_advantages(rewards, scale_std=cfg.get("scale_std", True)).view(-1).to(device)
        full_mask = torch.cat([inputs["attention_mask"], cmask], dim=1)

        # No learning signal (every group has identical rewards): skip the update entirely. The only gradient left
        # would be floating-point noise in the KL term, and Adam rescales tiny gradients to full-size steps, so
        # updating would be a random walk, and it also saves the reference and training forward passes.
        if cfg.get("skip_degenerate", True) and bool((rewards.std(1) < 1e-6).all()):
            rec = {"step": step + 1, "reward": rewards.mean().item(), "acc": acc, "format": fmt, "zero_var_groups": 1.0,
                   "completion_len": cmask.sum(1).float().mean().item(), "skipped": True, "sec": round(time.time() - t0, 1)}
            history.append(rec)
            logf.write(json.dumps(rec) + "\n"); logf.flush(); log(rec)
            continue

        # ---- reference log-probs
        with torch.no_grad(), reference_policy(model, has_ref_adapter):
            ref_lp = completion_logprobs(model, gen, full_mask, inputs["pixel_values"], inputs["image_grid_thw"], L)

        # ---- one on-policy update (old log-probs = current policy's, detached)
        model.train()
        N, stats = gen.shape[0], {}
        for idx in torch.arange(N).split(cfg.get("micro_batch", 4)):
            pv, grid = slice_visual(inputs["pixel_values"], inputs["image_grid_thw"], idx)
            lp = completion_logprobs(model, gen[idx], full_mask[idx], pv, grid, L)
            loss, stats = grpo_loss(lp, lp.detach(), ref_lp[idx], adv[idx], cmask[idx], clip=cfg.get("clip", 0.2),
                                    beta=cfg.get("beta", 0.02), length_norm=cfg.get("length_norm", "seq"))
            (loss * len(idx) / N).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.get("clip_grad", 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)

        rec = {"step": step + 1, "reward": rewards.mean().item(), "acc": acc, "format": fmt,
               "zero_var_groups": (rewards.std(1) < 1e-6).float().mean().item(),
               "completion_len": cmask.sum(1).float().mean().item(), "grad_norm": float(gn), **stats,
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        logf.write(json.dumps(rec) + "\n"); logf.flush(); log(rec)
        if eval_fn and cfg.get("eval_every") and (step + 1) % cfg["eval_every"] == 0:
            model.eval()
            log({"step": step + 1, "eval": eval_fn(model)})
        if cfg.get("save_every") and (step + 1) % cfg["save_every"] == 0 and hasattr(model, "save_pretrained"):
            model.save_pretrained(out / f"adapter_step{step + 1}")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out / "adapter_final")
    return history


def main():
    from peft import PeftModel

    from .eval import evaluate
    from .lora import apply_lora, trainable_summary
    from .model_utils import load_model, load_processor

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    torch.manual_seed(cfg["train"].get("seed", 0))
    processor = load_processor(cfg["model_path"])
    base = load_model(cfg["model_path"])
    init = cfg.get("init_adapter")
    if init:  # continue from an SFT adapter; keep a frozen copy as the KL reference
        model = PeftModel.from_pretrained(base, init, is_trainable=True)
        model.load_adapter(init, adapter_name="ref", is_trainable=False)
        model.set_adapter("default")
    else:
        model = apply_lora(base, **cfg.get("lora", {}))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(trainable_summary(model))
    train = make_split("train", cfg["data"]["n_train"], tasks=TASKS)
    val = make_split("val", cfg["data"]["n_eval"], tasks=TASKS)
    tcfg = dict(cfg["train"], out_dir=cfg["out_dir"])
    grpo_train(model, processor, train, tcfg, next(model.parameters()).device, has_ref_adapter=bool(init),
               eval_fn=lambda m: {k: v for k, v in evaluate(m, processor, val, batch=16).items() if k in ("accuracy", "format_rate")})


if __name__ == "__main__":
    main()
