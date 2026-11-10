import copy

import numpy as np
import pytest
import torch
from fakes import EOS, FakeProcessor, tiny_model

from vlm_lab.data import GenCollator, SFTCollator, build_messages, mask_labels
from vlm_lab.eval import evaluate, generate_texts
from vlm_lab.lora import LM_LINEAR_REGEX, apply_lora, lora_module_names, trainable_summary
from vlm_lab.sft import cosine_with_warmup, train_sft
from vlm_lab.tasks import TASKS, make_example, make_split
from vlm_lab.vision_utils import slice_visual


def decode(proc, ids):
    return proc.batch_decode(torch.tensor([ids]))[0]


def test_mask_labels_supervises_only_the_response_with_right_padding():
    ids = torch.tensor([[5, 6, 7, 8, 9, 0], [5, 6, 7, 8, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    lab = mask_labels(ids, mask, torch.tensor([3, 4]))
    assert lab.tolist() == [[-100, -100, -100, 8, 9, -100], [-100, -100, -100, -100, 9, 10]]


def test_sft_collator_labels_are_exactly_response_plus_end_token():
    proc = FakeProcessor()
    exs = [make_example("train", i) for i in range(4)]
    batch = SFTCollator(proc)(exs)
    assert set(batch) >= {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "labels"}
    for i, e in enumerate(exs):
        lab = batch["labels"][i]
        sup = lab[lab != -100].tolist()
        assert sup[-1] == EOS, "the end-of-turn token must be supervised so the model learns to stop"
        assert decode(proc, sup) == e.response()
        # no image/vision tokens and no prompt text are supervised
        assert not any(t in (250, 252, 253) for t in sup)
        first = int((lab != -100).nonzero()[0])
        assert batch["attention_mask"][i, :first].all()  # supervised region starts after the whole prompt
    # 4 images x 4 patches each, concatenated along dim 0
    assert batch["pixel_values"].shape == (16, 3 * 2 * 14 * 14) and batch["image_grid_thw"].tolist() == [[1, 2, 2]] * 4


def test_gen_collator_left_pads_and_restores_padding_side():
    proc = FakeProcessor()
    exs = [make_example("train", i) for i in range(5)]
    proc.tokenizer.padding_side = "right"
    b = GenCollator(proc)(exs)
    assert proc.tokenizer.padding_side == "right"  # restored
    am = b["attention_mask"]
    assert (am[:, -1] == 1).all(), "left padding: every row ends with a real token"
    assert am.sum(1).min() < am.shape[1], "prompts of different lengths were padded"
    ends = [decode(proc, row[m.bool()].tolist()) for row, m in zip(b["input_ids"], am)]
    assert all(s.endswith("\nA:") for s in ends)


def test_build_messages_has_image_and_prompt():
    m = build_messages(make_example("train", 1))
    assert m[0]["role"] == "user" and m[0]["content"][0] == {"type": "image"}
    assert "<answer>" in m[0]["content"][1]["text"]


def test_slice_visual_returns_the_right_rows():
    grid = torch.tensor([[1, 2, 2], [1, 4, 2], [1, 2, 4], [1, 2, 2]])
    counts = [4, 8, 8, 4]
    pv = torch.cat([torch.full((c, 3), float(i)) for i, c in enumerate(counts)])
    sub, g = slice_visual(pv, grid, [1, 3])
    assert sub.shape[0] == 12 and g.tolist() == [[1, 4, 2], [1, 2, 2]]
    assert (sub[:8] == 1).all() and (sub[8:] == 3).all()
    sub, g = slice_visual(pv, grid, torch.tensor([0, 2]))
    assert sub.shape[0] == 12 and (sub[:4] == 0).all() and (sub[4:] == 2).all()


# ---------------------------------------------------------------- LoRA
def test_lora_regex_hits_language_model_and_skips_vision():
    import re

    assert re.fullmatch(LM_LINEAR_REGEX, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(LM_LINEAR_REGEX, "model.layers.3.mlp.down_proj")          # older transformers layout
    assert not re.fullmatch(LM_LINEAR_REGEX, "model.visual.blocks.0.mlp.gate_proj")  # vision MLP has the same leaf names
    assert not re.fullmatch(LM_LINEAR_REGEX, "lm_head")


def test_lora_adapts_only_language_layers_and_freezes_everything_else():
    m = apply_lora(tiny_model(), r=4, alpha=8)
    names = lora_module_names(m)
    assert names and all("visual" not in n for n in names)
    assert all("language_model" in n for n in names)
    for n, p in m.named_parameters():
        if "lora_" in n:
            assert p.requires_grad
        else:
            assert not p.requires_grad, n
    s = trainable_summary(m)
    assert 0 < s["pct"] < 30


def test_sft_loop_overfits_a_small_batch_with_full_finetuning(tmp_path):
    """The training loop itself (collation, loss, optimizer, schedule) must be able to memorize 6 examples."""
    proc = FakeProcessor()
    model = tiny_model()
    cfg = {"batch": 3, "steps": 60, "lr": 1e-3, "warmup": 3, "log_every": 1000, "out_dir": str(tmp_path)}
    losses = train_sft(model, proc, make_split("train", 6, tasks=TASKS), cfg, "cpu", log=lambda *_: None)
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < 0.75 * np.mean(losses[:5]), (np.mean(losses[:5]), np.mean(losses[-5:]))


def test_sft_with_lora_learns_a_little_and_touches_only_lora_weights(tmp_path):
    """LoRA on a randomly initialized TINY model (frozen random embeddings and output head, 64 hidden units) is
    capacity-limited, so only a modest loss drop is expected here; this test is about WHICH weights move.
    (Real fine-tuning starts from a pretrained model where LoRA is plenty.)"""
    proc = FakeProcessor()
    model = apply_lora(tiny_model(), r=8, alpha=16)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cfg = {"batch": 3, "steps": 60, "lr": 1e-2, "warmup": 3, "log_every": 1000, "out_dir": str(tmp_path)}
    losses = train_sft(model, proc, make_split("train", 6, tasks=TASKS), cfg, "cpu", log=lambda *_: None)
    assert np.all(np.isfinite(losses))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.15, (np.mean(losses[:5]), np.mean(losses[-5:]))
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all("lora_" in k for k in changed), [k for k in changed if "lora_" not in k][:5]
    assert not any("visual" in k for k in changed)
    assert (tmp_path / "adapter").exists()


def test_cosine_schedule_shape():
    f = [cosine_with_warmup(s, 100, 10) for s in range(100)]
    assert f[0] < f[9] <= 1.0 and max(f) == pytest.approx(1.0, abs=0.02) and f[-1] < 0.01
    assert all(a >= b - 1e-9 for a, b in zip(f[10:], f[11:]))


# ---------------------------------------------------------------- generation / evaluation plumbing
def test_generate_texts_and_evaluate_run_end_to_end_on_the_tiny_model():
    proc = FakeProcessor()
    model = tiny_model().eval()
    exs = make_split("val", 6, tasks=TASKS)
    texts = generate_texts(model, proc, exs, max_new_tokens=5)
    assert len(texts) == 6 and all(isinstance(t, str) for t in texts)
    res = evaluate(model, proc, exs, batch=3, max_new_tokens=5)
    assert res["n"] == 6 and 0.0 <= res["accuracy"] <= 1.0 and set(res["per_task"]) == set(TASKS)
    assert len(res["scores"]) == 6 and res["accuracy_ci95"][0] <= res["accuracy"] <= res["accuracy_ci95"][1]
