import numpy as np
import pytest
import torch
from fakes import EOS, PAD, FakeProcessor, tiny_model

from vlm_lab import grpo as grpo_mod
from vlm_lab.data import GenCollator, to_device
from vlm_lab.grpo import completion_logprobs, completion_mask, grpo_train, reference_policy
from vlm_lab.grpo_math import group_advantages, grpo_loss, kl_k3
from vlm_lab.lora import apply_lora
from vlm_lab.tasks import TASKS, make_split


# ---------------------------------------------------------------- pure math (same checks as the SQL GRPO project)
def test_advantages_zero_mean_and_degenerate_group():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = group_advantages(r)
    assert torch.allclose(a.mean(1), torch.zeros(2), atol=1e-6) and torch.allclose(a[1], torch.zeros(4))


def test_kl_k3_nonneg_and_loss_directions_and_clipping():
    lp = torch.randn(4, 6)
    assert torch.allclose(kl_k3(lp, lp), torch.zeros(4, 6), atol=1e-7) and (kl_k3(lp, torch.randn(4, 6)) >= -1e-6).all()
    logp = torch.full((2, 5), -1.0, requires_grad=True)
    loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), torch.tensor([1.0, -1.0]), torch.ones(2, 5), beta=0.0)
    loss.backward()
    assert logp.grad[0].sum() < 0 < logp.grad[1].sum()
    old = torch.zeros(1, 3)
    far = torch.full((1, 3), 1.0, requires_grad=True)
    l2, st = grpo_loss(far, old, old, torch.tensor([1.0]), torch.ones(1, 3), beta=0.0)
    l2.backward()
    assert torch.count_nonzero(far.grad) == 0 and st["clip_frac"] == 1.0


def test_completion_mask_stops_after_first_stop_token():
    tail = torch.tensor([[7, 8, EOS, PAD, PAD], [7, 8, 9, 10, 11], [EOS, 5, 5, 5, 5]])
    m = completion_mask(tail, {EOS})
    assert m.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]


# ---------------------------------------------------------------- alignment on the real HF architecture
def _batch(n=3):
    proc = FakeProcessor()
    exs = make_split("train", n, tasks=TASKS)
    return proc, exs, to_device(GenCollator(proc)(exs), "cpu")


def test_greedy_generation_equals_argmax_of_teacher_forced_logits_with_left_padding():
    """If [prompt | completion] indexing, left padding, or the multimodal rope positions were inconsistent between
    generate() and a full forward pass, the log-probs GRPO trains on would be silently wrong."""
    proc, exs, inputs = _batch(4)
    L = inputs["input_ids"].shape[1]
    assert inputs["attention_mask"].sum(1).min() < L, "need genuinely different prompt lengths to test padding"
    model = tiny_model().eval()
    with torch.no_grad():
        gen = model.generate(**inputs, do_sample=False, max_new_tokens=6, pad_token_id=PAD, eos_token_id=299, use_cache=True)
        comp = gen[:, L:]
        full_mask = torch.cat([inputs["attention_mask"], torch.ones_like(comp)], dim=1)
        out = model(input_ids=gen, attention_mask=full_mask, pixel_values=inputs["pixel_values"],
                    image_grid_thw=inputs["image_grid_thw"])
    pred = out.logits[:, L - 1 : -1].argmax(-1)
    assert comp.shape == (4, 6) and torch.equal(pred, comp), (pred, comp)


def test_completion_logprobs_match_manual_gather_and_are_valid():
    proc, exs, inputs = _batch(3)
    L = inputs["input_ids"].shape[1]
    model = tiny_model().eval()
    with torch.no_grad():
        gen = model.generate(**inputs, do_sample=True, temperature=1.0, max_new_tokens=5, pad_token_id=PAD, eos_token_id=299)
        mask = torch.cat([inputs["attention_mask"], torch.ones_like(gen[:, L:])], 1)
        lp = completion_logprobs(model, gen, mask, inputs["pixel_values"], inputs["image_grid_thw"], L)
        logits = model(input_ids=gen, attention_mask=mask, pixel_values=inputs["pixel_values"],
                       image_grid_thw=inputs["image_grid_thw"]).logits.float()
    manual = torch.log_softmax(logits[:, L - 1 : -1], -1).gather(-1, gen[:, L:, None])[..., 0]
    assert torch.allclose(lp, manual, atol=1e-6) and (lp <= 1e-6).all() and lp.shape == (3, 5)


def test_reference_equals_policy_at_init_so_kl_starts_at_zero():
    """LoRA B starts at zero, so the adapted model equals the base model: the disabled-adapter reference must give
    identical log-probs. (A mismatch here would mean the reference is not what we think it is.)"""
    proc, exs, inputs = _batch(3)
    L = inputs["input_ids"].shape[1]
    model = apply_lora(tiny_model(), r=4, alpha=8).eval()
    with torch.no_grad():
        gen = model.generate(**inputs, do_sample=True, max_new_tokens=5, pad_token_id=PAD, eos_token_id=299)
        mask = torch.cat([inputs["attention_mask"], torch.ones_like(gen[:, L:])], 1)
        pol = completion_logprobs(model, gen, mask, inputs["pixel_values"], inputs["image_grid_thw"], L)
        with reference_policy(model, has_ref_adapter=False):
            ref = completion_logprobs(model, gen, mask, inputs["pixel_values"], inputs["image_grid_thw"], L)
    assert torch.allclose(pol, ref, atol=1e-6)


def test_second_adapter_reference_is_frozen_while_policy_trains():
    from peft import LoraConfig

    from vlm_lab.lora import LM_LINEAR_REGEX

    m = apply_lora(tiny_model(), r=4, alpha=8)
    m.add_adapter("ref", LoraConfig(r=4, lora_alpha=8, target_modules=LM_LINEAR_REGEX))
    m.set_adapter("default")
    pol_params = [p for n, p in m.named_parameters() if p.requires_grad]
    assert pol_params and all(".default." in n or "default" in n for n, p in m.named_parameters() if p.requires_grad)
    assert not any("ref" in n for n, p in m.named_parameters() if p.requires_grad and "lora_" in n)


# ---------------------------------------------------------------- the training loop, end to end
def test_grpo_runs_updates_only_lora_and_learns_a_synthetic_reward(monkeypatch, tmp_path):
    """A synthetic reward (completion decodes to >= 3 printable characters) stands in for the exact verifier,
    because a random tiny model essentially never writes a correct <answer>. The point is the mechanics: groups,
    advantages, masked log-probs, the update, and that policy-gradient steps actually raise reward."""
    monkeypatch.setattr(grpo_mod, "total_reward", lambda text, gold: 1.0 if len(text) >= 3 else 0.0)
    monkeypatch.setattr(grpo_mod, "accuracy_reward", lambda text, gold: 0.0)
    proc = FakeProcessor()
    model = apply_lora(tiny_model(), r=8, alpha=16)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cfg = {"prompts_per_step": 4, "group_size": 6, "max_new_tokens": 6, "steps": 40, "lr": 5e-3, "micro_batch": 8,
           "beta": 0.0, "temperature": 1.0, "seed": 0, "out_dir": str(tmp_path), "stop_ids": [299]}
    hist = grpo_train(model, proc, make_split("train", 32, tasks=TASKS), cfg, "cpu", log=lambda *_: None)
    updated = [h for h in hist if not h.get("skipped")]  # steps with no reward variance are skipped by design
    assert len(hist) == 40 and len(updated) >= 20
    assert all(np.isfinite(h["reward"]) for h in hist) and all(np.isfinite(h["grad_norm"]) for h in updated)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all("lora_" in k for k in changed)
    first, last = np.mean([h["reward"] for h in hist[:8]]), np.mean([h["reward"] for h in hist[-8:]])
    # across 8 different RNG states the reward rose from ~0.6-0.75 to 0.88-0.99 (rise 0.14-0.41); assert the robust part
    assert last >= 0.85 and last > first + 0.10, (first, last)
    assert (tmp_path / "adapter_final").exists() and (tmp_path / "log.jsonl").exists()


def test_grpo_with_no_reward_signal_leaves_the_policy_unchanged(monkeypatch, tmp_path):
    """All rewards equal -> zero advantages and zero KL: nothing should move (guards against a stray gradient)."""
    monkeypatch.setattr(grpo_mod, "total_reward", lambda text, gold: 0.0)
    proc = FakeProcessor()
    model = apply_lora(tiny_model(), r=4, alpha=8)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cfg = {"prompts_per_step": 2, "group_size": 3, "max_new_tokens": 4, "steps": 3, "lr": 1e-2, "micro_batch": 3,
           "beta": 0.02, "out_dir": str(tmp_path), "stop_ids": [299]}
    hist = grpo_train(model, proc, make_split("train", 8, tasks=TASKS), cfg, "cpu", log=lambda *_: None)
    assert all(h["zero_var_groups"] == 1.0 for h in hist)
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
