#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
M=Qwen/Qwen2.5-VL-3B-Instruct
$PY -m vlm_lab.grpo --config configs/grpo_base.yaml > results/grpo_base_train.log 2>&1
$PY -m vlm_lab.eval --model $M --adapter runs/grpo_base/adapter_final --splits val,ood --n 300 --batch 16 --out results/eval_C_grpo_base.json > results/eval_C.log 2>&1
$PY -m vlm_lab.grpo --config configs/grpo_sft.yaml > results/grpo_sft_train.log 2>&1
$PY -m vlm_lab.eval --model $M --adapter runs/grpo_sft/adapter_final --splits val,ood --n 300 --batch 16 --out results/eval_D_sft_grpo.json > results/eval_D.log 2>&1
echo done > results/GRPO_DONE
