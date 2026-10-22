#!/bin/bash
cd /Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering
export PYTHONPATH=src PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 ME_ATTENTION_RECORDED_WL=1
OUT=publications/me_attention/generated_recwl
for ds in collagen_pepsin lichens1 lichens2; do
  for m in meas_tokens ex_tokens_v2; do
    nice -n 5 .venv/bin/python experiments/me_attention/importance.py --dataset $ds --model $m --sweep recwl --repeat 0 --fold 0 --device mps --max-pixels 20000 --out $OUT/importance/${ds}_${m} || echo "FAILED importance $ds $m"
  done
done
echo IMPORTANCE_DONE
