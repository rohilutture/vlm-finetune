# Vision-language model fine-tuning: LoRA SFT and GRPO with an exact verifier

Status: **all four arms were run on a real Qwen2.5-VL-3B-Instruct on one RTX 5090 (single seed): zero-shot, LoRA SFT
(1,500 steps), GRPO from the base model (300 steps) and SFT then GRPO (150 steps). 48 CPU tests pass. The procedural
tasks turned out to be easy, so there is little headroom, and the results should be read as a ceiling-effect study.**

## Results (300 balanced `val` and 300 `ood` examples each, greedy decoding; `results/`, paired bootstrap in `results/paired_bootstrap.txt`)

| Arm | val accuracy | ood accuracy | mean output chars (ood) |
|---|---|---|---|
| A zero-shot | 90.0% | 86.3% | 120 |
| B LoRA SFT (1,500 steps) | 99.0% | 98.0% | 81 |
| C GRPO from base (300 steps) | 98.0% | 95.0% | 18 |
| D SFT then GRPO (150 steps) | 99.0% | 98.3% | 81 |

Paired differences on identical examples (95% bootstrap interval): SFT minus zero-shot +9.0 pts on val [+5.7, +12.3] and
+11.7 on ood [+7.3, +16.0]; GRPO-from-base minus zero-shot +8.0 [+5.0, +11.3] val and +8.7 [+5.0, +12.7] ood;
GRPO-from-base minus SFT -1.0 [-2.7, +0.3] val and **-3.0 [-5.7, -0.3] ood**; SFT+GRPO minus SFT exactly 0.0 on val and
+0.3 [-0.7, +1.7] on ood.

What this shows, and what it does not:

- Both SFT and GRPO from the base model close most of the gap to the verifier-checked answer; the weak spots of the base
  model were chart comparison (78% val, 70% ood) and spatial relations (82%, 84%).
- **SFT beat GRPO from the base model out of distribution** (98.0 vs 95.0%), with a small interval excluding zero;
  the hypothesis that RL generalizes better than SFT was not supported on this OOD split (new colours, more and smaller
  objects). The weakest OOD task for GRPO was colour counting (86%).
- **GRPO on top of SFT did nothing, by construction.** After SFT 99% of GRPO groups were all-correct, so the group-relative
  advantage is zero and the update is skipped (zero-variance groups 0.99); arm D is the SFT model with a little extra noise.
  This is a ceiling effect, not evidence that RL cannot improve a strong policy.
- GRPO from the base model shortened responses sharply (42 to 8 completion tokens on average; outputs are just
  `<answer>X</answer>`, so it dropped the base model's short explanations). Spot checks of outputs showed legitimate
  answers, not a format exploit, but some of the early accuracy gain is format compliance (format rate 0.865 to 0.989).
- Procedural synthetic tasks only; one seed; 3B model; 300 examples per split (a 3-point difference is near the noise
  floor, which is why the paired intervals matter). Nothing here says anything about real-world VQA benchmarks.

Times on one RTX 5090 (bf16, LoRA r=16): SFT about 2.2 s per optimizer step (55 min for 1,500); GRPO about 9 s per step
with 4 prompts x 8 samples (about 45 min for 300 steps).

Fine-tune Qwen2.5-VL-3B-Instruct on procedural visual-reasoning tasks whose answers are exact, so the reward is a
verifier rather than a learned model, in two stages: supervised fine-tuning with LoRA, then GRPO (reinforcement
learning from the verifier). Everything needed for the experiment is built and tested; the experiment itself has not run.

## The tasks (`vlm_lab/tasks.py`)

Generated from a seed, so reproducible and download-free. The answer is computed from the scene description, never
from the image. `python -m vlm_lab.preview` writes sample images (`out/preview/`).

| Task | Example | Answer |
|---|---|---|
| `count_shape` | "How many red circles are in the image?" | integer |
| `count_color` | "How many blue objects are in the image?" | integer |
| `relation` | "Is the green square left of the red circle?" (left/right/above/below) | yes / no |
| `chart_max` | "Which bar is the tallest?" | letter |
| `chart_value` | "What is the value of bar C?" (read off the axis) | integer |
| `chart_cmp` | "Which bar is shorter, B or D?" | letter |

Splits use disjoint seed ranges. `train`/`val`/`test` share one distribution (3-7 objects, six colours). **`ood` is
deliberately different**: 9-12 smaller objects and three colours (pink, cyan, brown) never seen in training, to measure
generalization instead of memorization. The SFT target is a short programmatic rationale followed by
`<answer>X</answer>`; the rationale quotes exact scene coordinates the model can only estimate, so judge the model by
the final answer, not by the rationale.

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Verifier | `verify.py` | extracts the last `<answer>` tag, normalizes, exact match; `total_reward` = 1.0 if correct else 0.1 for a well-formed answer, so format credit can never outrank correctness |
| Data | `data.py` | chat formatting; right-padded SFT batches with loss on the response only (including the end-of-turn token); left-padded generation batches |
| LoRA | `lora.py` | adapters on the language model's attention/MLP layers only; the vision tower and projector stay frozen |
| SFT | `sft.py` | LoRA fine-tuning loop (cosine schedule, grad accumulation) |
| GRPO | `grpo.py`, `grpo_math.py` | G samples per image+question, group-relative advantages, clipped surrogate, k3 KL to a frozen reference |
| Evaluation | `eval.py`, `stats.py` | greedy decoding, per-task accuracy with Wilson intervals, format rate, paired bootstrap for comparing systems |

## What is verified (CPU, `pytest`)

- **The images really encode the answers.** The tests re-derive each answer from the rendered *pixels alone*:
  connected-component counts by colour and by shape, object positions for the relation questions, and bar heights
  for charts, on both `train` and `ood`. This caught a real data bug: two same-colour shapes could touch at their
  corners (a square reaches r*sqrt(2) from its centre but I enforced only r), merging into one blob and making the
  labelled count wrong.
- Determinism, no scene leakage between splits, the OOD split really differs, objects never overlap, answers are
  balanced enough not to be guessable.
- **Tests run against the real `transformers` Qwen2.5-VL class** (random tiny weights, fake tokenizer/processor):
  the label mask supervises exactly the response; LoRA regex hits language-model layers and skips the similarly named
  vision layers; vision and base weights do not move during SFT; the SFT loop overfits a batch (full fine-tuning
  loss 5.5 to 3.3 in 60 steps).
- **GRPO alignment**: for a left-padded batch, greedy generation equals the argmax of the teacher-forced logits at
  every completion position, so the `[prompt | completion]` indexing and multimodal rope positions are consistent;
  reference log-probs equal policy log-probs at initialization (KL starts at 0); a second frozen adapter works as the
  reference.
- **GRPO learns**: with a synthetic reward standing in for the verifier (a random tiny model never writes a correct
  answer), mean reward rose from 0.57-0.76 to 0.88-0.99 across 8 RNG states, and only LoRA weights moved.

Bugs the tests found in the training code (now fixed, with tests): (1) the policy can sample the image-placeholder
token as a completion token, which crashes the next teacher-forced forward pass, so vision special tokens are
suppressed during sampling; (2) a step where every group has equal rewards produces only floating-point-noise
gradients, which Adam rescales to full-size random steps, so such steps are skipped. Test-side: a fake tokenizer that
could not encode a newline, and a too-optimistic loss-drop threshold for LoRA on a randomly initialized tiny model.

## Experiment design (arms A to D below were run; results above)

All four arms evaluated on 300 balanced `val` and 300 `ood` examples, greedy decoding:

| Arm | How |
|---|---|
| A zero-shot | base Qwen2.5-VL-3B-Instruct with the same prompt |
| B SFT | `configs/sft.yaml` |
| C GRPO from base | `configs/grpo.yaml` with `init_adapter: null` |
| D SFT then GRPO | `configs/grpo.yaml` (default) |

Report per-task accuracy with Wilson intervals and compare arms with the paired bootstrap on identical examples.
At 300 examples a 3-point difference is noise. Also watch the `ood` gap (does RL generalize better than SFT?), the
response length, and the format rate; and check reward hacking by reading samples, not just the reward curve.

## Run

```powershell
$env:PYTHONPATH = "."
.venv\Scripts\python.exe -m pytest -q                         # 48 tests, CPU, ~35 s
.venv\Scripts\python.exe -m vlm_lab.preview --out out/preview  # look at the tasks

# needs the model (hub id downloads ~7 GB; you must allow it, or point model_path at a local copy) and a free GPU:
.venv\Scripts\python.exe -m vlm_lab.eval --model Qwen/Qwen2.5-VL-3B-Instruct --splits val,ood --n 300   # arm A
.venv\Scripts\python.exe -m vlm_lab.sft  --config configs/sft.yaml
.venv\Scripts\python.exe -m vlm_lab.grpo --config configs/grpo.yaml
.venv\Scripts\python.exe -m vlm_lab.eval --model Qwen/Qwen2.5-VL-3B-Instruct --adapter runs/grpo/adapter_final
```

## Not verified / honest limits

- **(Superseded: a real model was run, see Results; the unverified items below were not separately checked.)** The real processor's chat template and image-token expansion, bf16 numerics, memory use
  (a 3B model in bf16 plus LoRA plus G=8 sampling on a 32 GB card is expected to fit but has not been measured), and
  generation speed are all untested. The fake processor mimics only the interface.
- The tasks are **procedural toys**: coloured shapes and simple bar charts, not a real VQA or chart benchmark. Say
  "procedural visual-reasoning tasks", never "chart understanding" or "VQA" results.
- The CPU "GRPO learns" test uses a synthetic reward; it proves the mechanics, not that the verifier reward will
  improve a real model.
- GRPO does one on-policy update per batch (old log-probs = current log-probs, detached); no multi-epoch PPO reuse.
- Everything is single-seed unless you run more seeds.

## Phrasing rules (once there are results)

- Name the model, the task suite (procedural), the split (val vs `ood`), n, greedy decoding, and the interval.
- Claim "LoRA SFT and GRPO with an exact verifier on a procedural visual-reasoning suite, evaluated in-distribution
  and out-of-distribution", not general VLM capability.
- Safe now: "built and unit-tested a VLM fine-tuning pipeline (LoRA SFT, GRPO) on the real Qwen2.5-VL architecture
  with answers verified from rendered pixels". Do not state any accuracy until arm A-D have run.
