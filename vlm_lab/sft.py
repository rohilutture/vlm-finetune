"""LoRA supervised fine-tuning on the procedural tasks (response-only loss).

    python -m vlm_lab.sft --config configs/sft.yaml

train_sft() works with any model whose forward returns `.loss`, which is how the CPU tests exercise it with a tiny
random Qwen2.5-VL.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import SFTCollator, to_device
from .tasks import TASKS, make_split


def cosine_with_warmup(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def train_sft(model, processor, examples, cfg: dict, device, eval_fn=None, log=print) -> list[float]:
    collator = SFTCollator(processor)
    loader = DataLoader(examples, batch_size=cfg["batch"], shuffle=True, collate_fn=collator,
                        generator=torch.Generator().manual_seed(cfg.get("seed", 0)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0))
    accum, total = cfg.get("accum", 1), cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_with_warmup(s, total, cfg.get("warmup", 10)))
    out = Path(cfg.get("out_dir", "runs/sft"))
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "log.jsonl").open("a")
    losses, it, t0, step = [], iter(()), time.time(), 0

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    model.train()
    while step < total:
        acc_loss = 0.0
        for _ in range(accum):
            batch = to_device(next_batch(), device)
            loss = model(**batch).loss / accum
            loss.backward()
            acc_loss += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, cfg.get("clip", 1.0))
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        losses.append(acc_loss)
        if step % cfg.get("log_every", 10) == 0 or step == total:
            rec = {"step": step, "loss": sum(losses[-10:]) / len(losses[-10:]), "grad_norm": float(gn),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            logf.write(json.dumps(rec) + "\n"); logf.flush(); log(rec)
        if eval_fn and cfg.get("eval_every") and step % cfg["eval_every"] == 0:
            model.eval()
            log({"step": step, "eval": eval_fn(model)})
            model.train()
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out / "adapter")
    return losses


def main():
    from .eval import evaluate
    from .lora import apply_lora, trainable_summary
    from .model_utils import load_model, load_processor

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = yaml.safe_load(Path(ap.parse_args().config).read_text())
    torch.manual_seed(cfg["train"].get("seed", 0))
    processor = load_processor(cfg["model_path"])
    model = load_model(cfg["model_path"])
    model = apply_lora(model, **cfg.get("lora", {}))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(trainable_summary(model))
    train = make_split("train", cfg["data"]["n_train"], tasks=TASKS)
    val = make_split("val", cfg["data"]["n_eval"], tasks=TASKS)
    tcfg = dict(cfg["train"], out_dir=cfg["out_dir"])
    train_sft(model, processor, train, tcfg, next(model.parameters()).device,
              eval_fn=lambda m: {k: v for k, v in evaluate(m, processor, val, batch=16).items() if k in ("accuracy", "format_rate")})


if __name__ == "__main__":
    main()
