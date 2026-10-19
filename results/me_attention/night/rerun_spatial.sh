#!/bin/bash
cd /Users/narekmeloyan/PycharmProjects/spectral-select-attention
sleep 30
while pgrep -f "me_attention.harness --sweep night" >/dev/null; do sleep 60; done
echo "rerun lichens1/spatial_attn start $(date)" >> results/me_attention/night/RUN_INFO.md
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 PYTHONPATH=src
/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/.venv/bin/python -m me_attention.harness --sweep night \
  --datasets lichens1 --models spatial_attn --repeats lichens1=1 --budget 3 --max-epochs 25 --device mps --workers 2 --gpu-workers 2 --spatial-cap 2000
echo "rerun lichens1/spatial_attn end $(date)" >> results/me_attention/night/RUN_INFO.md
