"""Procedural visual-reasoning tasks with exact, verifiable answers.

Every example is generated from a seed, so it is reproducible, needs no dataset download, and its answer is known
by construction (computed from the scene description, never from the image).

Tasks
  count_shape  "How many red circles are in the image?"            -> integer
  count_color  "How many blue objects are in the image?"           -> integer
  relation     "Is the green square to the left of the red circle?" -> yes / no   (left of / right of / above / below)
  chart_max    "Which bar is the tallest?"                          -> letter
  chart_value  "What is the value of bar C?"                        -> integer (read off the axis)
  chart_cmp    "Which bar is shorter, B or D?"                      -> letter

Splits (disjoint seed ranges)
  train / val / test : the same distribution (3-7 objects, six colors, sparse scenes)
  ood                : harder and different: 9-12 smaller objects and THREE COLORS NEVER SEEN IN TRAINING, to measure
                       generalization instead of memorization.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

COLORS = {
    "red": (220, 40, 40), "green": (40, 170, 60), "blue": (50, 80, 220),
    "yellow": (240, 210, 40), "purple": (140, 60, 180), "orange": (240, 140, 30),
}
OOD_COLORS = {"pink": (240, 120, 180), "cyan": (40, 200, 210), "brown": (140, 90, 50)}
SHAPES = ("circle", "square", "triangle")
TASKS = ("count_shape", "count_color", "relation", "chart_max", "chart_value", "chart_cmp")
SEED_BASE = {"train": 0, "val": 10_000_000, "test": 20_000_000, "ood": 30_000_000}

INSTRUCTION = ("Think step by step briefly, then give the final answer inside <answer></answer> tags "
               "(a number, yes/no, or a letter).")


def rgb(color: str):
    return COLORS.get(color) or OOD_COLORS[color]


@dataclass
class Example:
    id: str
    split: str
    task: str
    question: str
    answer: str
    scene: dict
    rationale: str = ""
    meta: dict = field(default_factory=dict)

    def image(self) -> Image.Image:
        return render(self.scene)

    def prompt(self) -> str:
        return f"{self.question} {INSTRUCTION}"

    def response(self) -> str:
        """Target text for supervised fine-tuning: short programmatic reasoning, then the tagged answer."""
        return f"{self.rationale} <answer>{self.answer}</answer>"


# ----------------------------------------------------------------------------- shape scenes
def _place_objects(rng: random.Random, n: int, size: int, rmin: int, rmax: int, colors: list[str]) -> list[dict]:
    objs: list[dict] = []
    for _ in range(n):
        for _attempt in range(200):
            r = rng.randint(rmin, rmax)
            x, y = rng.randint(r + 4, size - r - 4), rng.randint(r + 4, size - r - 4)
            # squares/triangles reach r*sqrt(2) from their center, so the safe gap uses 1.42*r: otherwise two
            # same-colored shapes can touch at the corners and merge into one blob (then the count is ambiguous)
            if all(math.hypot(x - o["x"], y - o["y"]) > 1.42 * (r + o["r"]) + 4 for o in objs):
                objs.append({"shape": rng.choice(SHAPES), "color": rng.choice(colors), "x": x, "y": y, "r": r})
                break
    return objs


def _shape_scene(rng: random.Random, split: str) -> dict:
    if split == "ood":
        n, rmin, rmax = rng.randint(9, 12), 9, 13
        colors = list(COLORS) + list(OOD_COLORS)
    else:
        n, rmin, rmax = rng.randint(3, 7), 12, 20
        colors = list(COLORS)
    bg = rng.randint(238, 252)
    return {"kind": "shapes", "size": 224, "bg": bg, "objects": _place_objects(rng, n, 224, rmin, rmax, colors)}


def _describe(o: dict) -> str:
    return f"{o['color']} {o['shape']}"


def _unique_objects(objs: list[dict]) -> list[dict]:
    keys = [(o["color"], o["shape"]) for o in objs]
    return [o for o, k in zip(objs, keys) if keys.count(k) == 1]


def _plural(shape: str) -> str:
    return shape + "s"


def _make_count_shape(rng, split):
    scene = _shape_scene(rng, split)
    objs = scene["objects"]
    if objs and rng.random() < 0.8:
        t = rng.choice(objs)
        color, shape = t["color"], t["shape"]
    else:
        pool = list(COLORS) + (list(OOD_COLORS) if split == "ood" else [])
        color, shape = rng.choice(pool), rng.choice(SHAPES)
    n = sum(1 for o in objs if o["color"] == color and o["shape"] == shape)
    where = [f"({o['x']},{o['y']})" for o in objs if o["color"] == color and o["shape"] == shape]
    rat = (f"Looking for {color} {_plural(shape)}: " + (", ".join(where) if where else "none found") + f". That is {n}.")
    return scene, f"How many {color} {_plural(shape)} are in the image?", str(n), rat


def _make_count_color(rng, split):
    scene = _shape_scene(rng, split)
    objs = scene["objects"]
    pool = sorted({o["color"] for o in objs}) or list(COLORS)
    color = rng.choice(pool if rng.random() < 0.85 else list(COLORS) + (list(OOD_COLORS) if split == "ood" else []))
    hits = [o for o in objs if o["color"] == color]
    rat = f"{color.capitalize()} objects: " + (", ".join(f"{o['shape']} at ({o['x']},{o['y']})" for o in hits) if hits else "none") + f". That is {len(hits)}."
    return scene, f"How many {color} objects are in the image?", str(len(hits)), rat


def _make_relation(rng, split):
    for _ in range(100):
        scene = _shape_scene(rng, split)
        uniq = _unique_objects(scene["objects"])
        if len(uniq) < 2:
            continue
        a, b = rng.sample(uniq, 2)
        rel = rng.choice(["left of", "right of", "above", "below"])
        axis = "x" if rel in ("left of", "right of") else "y"
        if abs(a[axis] - b[axis]) < 10:  # too close to call by eye: resample
            continue
        truth = {"left of": a["x"] < b["x"], "right of": a["x"] > b["x"], "above": a["y"] < b["y"], "below": a["y"] > b["y"]}[rel]
        pa, pb = (a["x"], b["x"]) if axis == "x" else (a["y"], b["y"])
        op = "<" if pa < pb else ">"
        rat = (f"The {_describe(a)} is at {axis}={pa} and the {_describe(b)} is at {axis}={pb}. {pa} {op} {pb}, "
               f"so the {_describe(a)} is {'' if truth else 'not '}{rel} the {_describe(b)}.")
        return scene, f"Is the {_describe(a)} {rel} the {_describe(b)}?", "yes" if truth else "no", rat
    raise RuntimeError("could not build a relation scene")


# ----------------------------------------------------------------------------- bar charts
def _chart_scene(rng: random.Random) -> dict:
    k = rng.randint(4, 6)
    values = rng.sample(range(1, 10), k)  # distinct, so the tallest/shortest/ordering are unambiguous
    return {"kind": "chart", "size": 336, "bg": rng.randint(240, 255), "values": values,
            "labels": [chr(ord("A") + i) for i in range(k)], "value_labels": rng.random() < 0.3,
            "color": rng.choice(list(COLORS))}


def _make_chart_max(rng, split):
    scene = _chart_scene(rng)
    v, lab = scene["values"], scene["labels"]
    i = max(range(len(v)), key=lambda j: v[j])
    rat = "Bar values: " + ", ".join(f"{l}={x}" for l, x in zip(lab, v)) + f". The largest is {lab[i]}."
    return scene, "Which bar is the tallest?", lab[i], rat


def _make_chart_value(rng, split):
    scene = _chart_scene(rng)
    v, lab = scene["values"], scene["labels"]
    i = rng.randrange(len(v))
    rat = f"Reading bar {lab[i]} against the axis gives {v[i]}."
    return scene, f"What is the value of bar {lab[i]}?", str(v[i]), rat


def _make_chart_cmp(rng, split):
    scene = _chart_scene(rng)
    v, lab = scene["values"], scene["labels"]
    i, j = rng.sample(range(len(v)), 2)
    short = i if v[i] < v[j] else j
    rat = f"Bar {lab[i]} is {v[i]} and bar {lab[j]} is {v[j]}, so {lab[short]} is shorter."
    return scene, f"Which bar is shorter, {lab[i]} or {lab[j]}?", lab[short], rat


_MAKERS = {"count_shape": _make_count_shape, "count_color": _make_count_color, "relation": _make_relation,
           "chart_max": _make_chart_max, "chart_value": _make_chart_value, "chart_cmp": _make_chart_cmp}


def make_example(split: str, index: int, task: str | None = None) -> Example:
    """Deterministic: the same (split, index, task) always yields the same example."""
    rng = random.Random(SEED_BASE[split] + index)
    t = task or rng.choice(TASKS)
    scene, q, a, rat = _MAKERS[t](rng, split)
    return Example(id=f"{split}-{index}", split=split, task=t, question=q, answer=a, scene=scene, rationale=rat)


def make_split(split: str, n: int, tasks: tuple[str, ...] | None = None, start: int = 0) -> list[Example]:
    if tasks is None:
        return [make_example(split, start + i) for i in range(n)]
    return [make_example(split, start + i, tasks[i % len(tasks)]) for i in range(n)]


def scene_key(ex: Example) -> str:
    return json.dumps(ex.scene, sort_keys=True) + "|" + ex.question


# ----------------------------------------------------------------------------- rendering
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # old Pillow without sized default font
        return ImageFont.load_default()


def render(scene: dict) -> Image.Image:
    if scene["kind"] == "shapes":
        return _render_shapes(scene)
    return _render_chart(scene)


def _render_shapes(scene: dict) -> Image.Image:
    s = scene["size"]
    im = Image.new("RGB", (s, s), (scene["bg"],) * 3)
    d = ImageDraw.Draw(im)
    for o in scene["objects"]:
        x, y, r, c = o["x"], o["y"], o["r"], rgb(o["color"])
        if o["shape"] == "circle":
            d.ellipse([x - r, y - r, x + r, y + r], fill=c)
        elif o["shape"] == "square":
            d.rectangle([x - r, y - r, x + r, y + r], fill=c)
        else:
            d.polygon([(x, y - r), (x - r, y + r), (x + r, y + r)], fill=c)
    return im


def chart_geometry(scene: dict) -> dict:
    s = scene["size"]
    left, bottom, top, right = 44, s - 40, 24, s - 16
    k = len(scene["values"])
    slot = (right - left) / k
    return {"left": left, "bottom": bottom, "top": top, "right": right, "slot": slot,
            "unit": (bottom - top) / 10.0, "bar_w": slot * 0.6}


def _render_chart(scene: dict) -> Image.Image:
    s = scene["size"]
    g = chart_geometry(scene)
    im = Image.new("RGB", (s, s), (scene["bg"],) * 3)
    d = ImageDraw.Draw(im)
    f = _font(13)
    for t in range(0, 11):  # gridlines + tick labels 0..10
        y = g["bottom"] - t * g["unit"]
        d.line([(g["left"], y), (g["right"], y)], fill=(200, 200, 200), width=1)
        d.text((g["left"] - 24, y - 7), str(t), fill=(40, 40, 40), font=f)
    col = rgb(scene["color"])
    for i, (v, lab) in enumerate(zip(scene["values"], scene["labels"])):
        cx = g["left"] + g["slot"] * (i + 0.5)
        top = g["bottom"] - v * g["unit"]
        d.rectangle([cx - g["bar_w"] / 2, top, cx + g["bar_w"] / 2, g["bottom"]], fill=col)
        d.text((cx - 4, g["bottom"] + 8), lab, fill=(20, 20, 20), font=f)
        if scene["value_labels"]:
            d.text((cx - 4, top - 15), str(v), fill=(20, 20, 20), font=f)
    d.line([(g["left"], g["top"]), (g["left"], g["bottom"])], fill=(60, 60, 60), width=2)
    d.line([(g["left"], g["bottom"]), (g["right"], g["bottom"])], fill=(60, 60, 60), width=2)
    return im
