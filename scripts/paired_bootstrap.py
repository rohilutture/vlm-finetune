"""Paired bootstrap of accuracy differences between arms on the identical eval examples (results/eval_*.json)."""
import json, random
arms = {"A zero-shot": "results/eval_A_zeroshot.json", "B SFT": "results/eval_B_sft.json",
        "C GRPO from base": "results/eval_C_grpo_base.json", "D SFT+GRPO": "results/eval_D_sft_grpo.json"}
data = {k: json.load(open(v)) for k, v in arms.items()}
rng = random.Random(0)
def paired(a, b, split, B=10000):
    x, y = data[a][split]["scores"], data[b][split]["scores"]
    d = [float(p) - float(q) for p, q in zip(x, y)]
    n = len(d); means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return sum(d) / n, means[int(.025 * B)], means[int(.975 * B)]
print("n per split:", len(data["A zero-shot"]["val"]["scores"]))
for split in ("val", "ood"):
    print(f"== {split}: accuracy", {k: round(data[k][split]["accuracy"], 3) for k in arms})
    for a, b in [("B SFT", "A zero-shot"), ("C GRPO from base", "A zero-shot"), ("C GRPO from base", "B SFT"), ("D SFT+GRPO", "B SFT")]:
        m, lo, hi = paired(a, b, split)
        print(f"  {a} minus {b}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
