#!/bin/bash
# Chained: wait for the main 'night' run, then extend spectral repeats on pepsin + lichens2 (all 12 spectral models).
cd /Users/narekmeloyan/PycharmProjects/spectral-select-attention
while pgrep -f "me_attention.harness --sweep night" >/dev/null; do sleep 60; done
echo "extension start $(date)" >> results/me_attention/night/RUN_INFO.md
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 PYTHONPATH=src
/Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering/.venv/bin/python -m me_attention.harness --sweep night   --datasets collagen_pepsin,lichens2   --models lr,rbfsvm,rf,hgb,knn,mlp,cnn1d,spectralformer,smartattention_v0,ex_tokens_v2,hybrid_conv_attn,meas_tokens   --repeats collagen_pepsin=5,lichens2=5 --budget 3 --max-epochs 25 --device mps --workers 8 --gpu-workers 3 --order spectral-first
echo "extension end $(date)" >> results/me_attention/night/RUN_INFO.md
