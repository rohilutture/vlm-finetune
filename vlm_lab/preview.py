"""Write a few example images + their questions/answers so you can look at the tasks.
    python -m vlm_lab.preview --out out/preview --n 4"""
import argparse
import json
from pathlib import Path

from .tasks import TASKS, make_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/preview")
    ap.add_argument("--n", type=int, default=4, help="examples per task per split")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    for split in ("train", "ood"):
        for task in TASKS:
            for i, e in enumerate(make_split(split, a.n, tasks=(task,))):
                name = f"{split}_{task}_{i}.png"
                e.image().save(out / name)
                meta.append({"file": name, "question": e.question, "answer": e.answer, "response": e.response()})
    (out / "examples.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {len(meta)} images to {out}")


if __name__ == "__main__":
    main()
