"""Answer extraction and rewards. All rewards are exact and deterministic.

The model must finish with  <answer>X</answer>.  We take the LAST well-formed tag (so a model that rambles and
then answers is scored on its final answer) and normalize it: integers as plain integers, yes/no lower case,
bar letters upper case.
"""
from __future__ import annotations

import re

_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_INT = re.compile(r"^[+-]?\d+$")


def extract_answer(text: str) -> str | None:
    """Content of the last <answer>...</answer>, normalized; None if there is no well-formed tag."""
    m = _TAG.findall(text)
    if not m:
        return None
    return normalize(m[-1])


def normalize(s: str) -> str:
    s = s.strip().strip(".").strip()
    if _INT.match(s):
        return str(int(s))
    low = s.lower()
    if low in ("yes", "no"):
        return low
    if len(s) == 1 and s.isalpha():
        return s.upper()
    return low


def has_single_answer_tag(text: str) -> bool:
    return len(_TAG.findall(text)) == 1 and text.strip().lower().endswith("</answer>")


def accuracy_reward(text: str, gold: str) -> float:
    pred = extract_answer(text)
    return 1.0 if pred is not None and pred == normalize(gold) else 0.0


def format_reward(text: str) -> float:
    return 1.0 if has_single_answer_tag(text) else 0.0


def total_reward(text: str, gold: str) -> float:
    """Correctness dominates. Format credit exists only to bootstrap the tag format and can never outrank a
    correct answer: a wrong but well-formed answer gets 0.1, a correct one gets 1.0 (format included)."""
    acc = accuracy_reward(text, gold)
    if acc == 1.0:
        return 1.0
    return 0.1 * format_reward(text)
