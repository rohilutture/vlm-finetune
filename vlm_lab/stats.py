"""Confidence intervals for accuracy comparisons. With a few hundred eval examples, differences of a couple of
points are noise; report intervals, and use the paired bootstrap when comparing two systems on the SAME examples."""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_bootstrap_diff(a: list[float], b: list[float], iters: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Mean(a - b) over the same examples with a 95 percent bootstrap interval: (mean, lo, hi)."""
    assert len(a) == len(b) and a, "need paired, non-empty score lists"
    rng = random.Random(seed)
    n = len(a)
    d = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return (sum(d) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1])
