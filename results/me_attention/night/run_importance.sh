#!/bin/bash
cd /Users/narekmeloyan/PycharmProjects/spectral-select-attention
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 PYTHONPATH=src
PY=/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/.venv/bin/python
OUT=publications/me_attention/generated/importance
for ds in collagen_pepsin lichens1 lichens2; do
  for m in ex_tokens_v2 meas_tokens; do
    echo "== $ds $m $(date)"
    nice -n 5 $PY experiments/me_attention/importance.py --dataset $ds --model $m --sweep night --repeat 0 --fold 0 --device mps --max-pixels 20000 --out $OUT/${ds}_${m} || echo "FAILED $ds $m"
  done
done
echo "done $(date)"
