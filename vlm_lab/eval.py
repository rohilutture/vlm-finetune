"""Generation and evaluation. Greedy by default so results are reproducible; accuracy is exact match of the final
<answer> tag (see verify.py), reported per task with Wilson intervals, plus format-compliance and response length.

    python -m vlm_lab.eval --model Qwen/Qwen2.5-VL-3B-Instruct [--adapter runs/sft/adapter] --splits val,ood --n 300
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import GenCollator, to_device
from .stats import wilson
from .tasks import TASKS, Example, make_split
from .verify import accuracy_reward, format_reward


@torch.no_grad()
def generate_texts(model, processor, examples: list[Example], max_new_tokens: int = 160, temperature: float = 0.0,
                   device=None) -> list[str]:
    device = device or next(model.parameters()).device
    inputs = to_device(GenCollator(processor)(examples), device)
    L = inputs["input_ids"].shape[1]
    tok = processor.tokenizer
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=1.0)
    else:
        kw.update(do_sample=False)
    out = model.generate(**inputs, **kw)
    return processor.batch_decode(out[:, L:], skip_special_tokens=True)


def evaluate(model, processor, examples: list[Example], batch: int = 16, max_new_tokens: int = 160, device=None) -> dict:
    model.eval()
    texts: list[str] = []
    for i in range(0, len(examples), batch):
        texts += generate_texts(model, processor, examples[i:i + batch], max_new_tokens, 0.0, device)
    scores = [accuracy_reward(t, e.answer) for t, e in zip(texts, examples)]
    by_task: dict[str, list[float]] = defaultdict(list)
    for s, e in zip(scores, examples):
        by_task[e.task].append(s)
    res = {"n": len(examples), "accuracy": sum(scores) / len(scores),
           "format_rate": sum(format_reward(t) for t in texts) / len(texts),
           "mean_chars": sum(len(t) for t in texts) / len(texts), "scores": scores, "texts": texts}
    k, n = int(sum(scores)), len(scores)
    res["accuracy_ci95"] = wilson(k, n)
    res["per_task"] = {t: {"n": len(v), "accuracy": sum(v) / len(v), "ci95": wilson(int(sum(v)), len(v))}
                       for t, v in sorted(by_task.items())}
    return res


def summarize(res: dict) -> str:
    lo, hi = res["accuracy_ci95"]
    lines = [f"overall {res['accuracy']:.3f} [{lo:.3f}, {hi:.3f}] n={res['n']}  format {res['format_rate']:.2f}  "
             f"mean chars {res['mean_chars']:.0f}"]
    for t, r in res["per_task"].items():
        lines.append(f"  {t:12s} {r['accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] n={r['n']}")
    return "\n".join(lines)


def main():
    from .model_utils import load_model, load_processor

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--splits", default="val,ood")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()
    processor = load_processor(a.model)
    model = load_model(a.model)
    if a.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a.adapter)
    results = {}
    for split in a.splits.split(","):
        ex = make_split(split, a.n, tasks=TASKS)  # balanced across tasks
        res = evaluate(model, processor, ex, a.batch, a.max_new_tokens)
        print(f"== {split}\n{summarize(res)}")
        results[split] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
