#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
$PY -m vlm_lab.sft --config configs/sft.yaml > results/sft_train.log 2>&1
$PY -m vlm_lab.eval --model Qwen/Qwen2.5-VL-3B-Instruct --adapter runs/sft/adapter --splits val,ood --n 300 --batch 16 --out results/eval_B_sft.json > results/eval_B.log 2>&1
echo done > results/SFT_DONE
