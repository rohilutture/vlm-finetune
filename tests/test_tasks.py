"""Tasks must be deterministic, internally consistent, and (the important one) decodable from the rendered PIXELS:
if the image does not actually encode the answer, a model could never learn the task."""
import numpy as np
import pytest

from vlm_lab.tasks import (COLORS, OOD_COLORS, SEED_BASE, TASKS, chart_geometry, make_example, make_split, render,
                           rgb, scene_key)

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def blobs(img, color, tol=6):
    """Connected components of pixels matching `color` -> list of (cx, cy, area, kind), where kind is classified
    from the blob's OWN bounding box: fill ratio ~0.79 circle, ~1.0 square, ~0.5 triangle."""
    a = np.asarray(img).astype(int)
    mask = (np.abs(a - np.array(rgb(color))).max(axis=2) <= tol)
    seen = np.zeros_like(mask)
    out = []
    H, W = mask.shape
    for y in range(H):
        for x in range(W):
            if mask[y, x] and not seen[y, x]:
                stack, pts = [(y, x)], []
                seen[y, x] = True
                while stack:
                    cy, cx = stack.pop()
                    pts.append((cy, cx))
                    for dx, dy in DIRS:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                if len(pts) >= 20:  # ignore anti-aliasing specks
                    ys, xs = zip(*pts)
                    bw, bh = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
                    ratio = len(pts) / (bw * bh)
                    kind = "square" if ratio > 0.9 else "circle" if ratio > 0.65 else "triangle"
                    out.append((float(np.mean(xs)), float(np.mean(ys)), len(pts), kind))
    return out


def test_deterministic_and_splits_use_disjoint_seed_ranges():
    a, b = make_example("train", 7), make_example("train", 7)
    assert a.question == b.question and a.answer == b.answer and a.scene == b.scene
    assert a.question != make_example("train", 8).question or a.scene != make_example("train", 8).scene
    bases = sorted(SEED_BASE.values())
    assert all(y - x >= 10_000_000 for x, y in zip(bases, bases[1:]))


def test_no_scene_leakage_across_splits():
    keys = {s: {scene_key(e) for e in make_split(s, 400)} for s in ("train", "val", "test", "ood")}
    for x in keys:
        for y in keys:
            if x < y:
                assert not (keys[x] & keys[y]), f"{x} and {y} share examples"


def test_all_tasks_appear_and_answers_are_well_formed():
    ex = make_split("train", 600)
    assert {e.task for e in ex} == set(TASKS)
    for e in ex:
        if e.task.startswith("count") or e.task == "chart_value":
            assert e.answer.isdigit()
        elif e.task == "relation":
            assert e.answer in ("yes", "no")
        else:
            assert len(e.answer) == 1 and e.answer.isalpha() and e.answer.isupper()
        assert e.response().endswith(f"<answer>{e.answer}</answer>")


def test_answers_are_balanced_enough_to_not_be_guessable():
    ex = make_split("train", 1200)
    rel = [e.answer for e in ex if e.task == "relation"]
    assert 0.35 < rel.count("yes") / len(rel) < 0.65, "relation answers heavily skewed"
    counts = [int(e.answer) for e in ex if e.task == "count_shape"]
    assert len(set(counts)) >= 4 and counts.count(0) / len(counts) < 0.35


def test_ood_split_really_differs():
    tr, ood = make_split("train", 300), make_split("ood", 300)
    n = lambda es: np.mean([len(e.scene["objects"]) for e in es if e.scene["kind"] == "shapes"])  # noqa: E731
    assert n(ood) > n(tr) + 3
    train_colors = {o["color"] for e in tr for o in e.scene.get("objects", [])}
    ood_colors = {o["color"] for e in ood for o in e.scene.get("objects", [])}
    assert not (train_colors & set(OOD_COLORS)) and ood_colors & set(OOD_COLORS)


def test_objects_never_overlap():
    for e in make_split("ood", 200) + make_split("train", 200):
        objs = e.scene.get("objects", [])
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                assert ((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5 > a["r"] + b["r"]


# ---- the image must encode the answer ------------------------------------------------------------
@pytest.mark.parametrize("split", ["train", "ood"])
def test_count_color_answer_is_recoverable_from_pixels(split):
    checked = 0
    for e in make_split(split, 400, tasks=("count_color",)):
        color = e.question.split(" ")[2]
        found = len(blobs(e.image(), color))
        assert found == int(e.answer), (e.question, e.answer, found)
        checked += 1
    assert checked == 400


def test_count_shape_answer_is_recoverable_from_pixels():
    for e in make_split("train", 200, tasks=("count_shape",)):
        words = e.question.split(" ")  # "How many <color> <shape>s are in the image?"
        color, shape = words[2], words[3].rstrip("s")
        objs = [o for o in e.scene["objects"] if o["color"] == color]
        found = blobs(e.image(), color)
        assert len(found) == len(objs)
        n = sum(1 for (_, _, _, kind) in found if kind == shape)
        assert n == int(e.answer), (e.question, e.answer, n)


def test_relation_answer_is_recoverable_from_pixels():
    for e in make_split("train", 300, tasks=("relation",)):
        q = e.question[len("Is the "):-1]  # "<color> <shape> <rel> the <color> <shape>"
        for rel in ("left of", "right of", "above", "below"):
            if f" {rel} the " in q:
                left, right = q.split(f" {rel} the ")
                break
        (ca, sa), (cb, sb) = left.split(" "), right.split(" ")
        img = e.image()

        def locate(color, shape):  # the (color, shape) pair is unique in the scene, but the color may repeat on other shapes
            hits = [(cx, cy) for (cx, cy, _, kind) in blobs(img, color) if kind == shape]
            assert len(hits) == 1, (e.question, color, shape, hits)
            return hits[0]

        A, B = locate(ca, sa), locate(cb, sb)
        truth = {"left of": A[0] < B[0], "right of": A[0] > B[0], "above": A[1] < B[1], "below": A[1] > B[1]}[rel]
        assert ("yes" if truth else "no") == e.answer, (e.question, e.answer, A, B)


def test_chart_values_are_recoverable_from_bar_heights():
    for e in make_split("train", 200, tasks=("chart_value", "chart_max", "chart_cmp")):
        sc = e.scene
        g = chart_geometry(sc)
        img = np.asarray(e.image()).astype(int)
        target = np.array(rgb(sc["color"]))
        measured = []
        for i in range(len(sc["values"])):
            cx = int(g["left"] + g["slot"] * (i + 0.5))
            col = (np.abs(img[:, cx] - target).max(axis=1) <= 6)
            ys = np.nonzero(col)[0]
            measured.append(round((g["bottom"] - ys.min()) / g["unit"]))
        assert measured == sc["values"], (measured, sc["values"])
        if e.task == "chart_max":
            assert sc["labels"][int(np.argmax(measured))] == e.answer
        if e.task == "chart_value":
            lab = e.question.split()[-1].rstrip("?")
            assert str(measured[sc["labels"].index(lab)]) == e.answer
        if e.task == "chart_cmp":
            q = e.question.replace("Which bar is shorter, ", "").rstrip("?")
            x, y = q.split(" or ")
            short = x if measured[sc["labels"].index(x)] < measured[sc["labels"].index(y)] else y
            assert short == e.answer


def test_render_sizes_and_chart_has_axis_labels_visible():
    s = make_example("train", 3, "count_shape").image()
    c = make_example("train", 3, "chart_max").image()
    assert s.size == (224, 224) and c.size == (336, 336) and s.mode == "RGB"
    # the chart has dark text/axis pixels (tick labels and axes), i.e. text actually rendered
    assert (np.asarray(c).sum(axis=2) < 200).sum() > 200
