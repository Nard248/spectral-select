#!/bin/bash
# Noise check: rerun the 19 Oct configuration (shifted labels, ME_ATTENTION_RECORDED_WL unset) for the spectral attention
# models with identical seeds, to separate run-to-run (MPS) variation from the effect of the recorded wavelengths.
cd /Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2 PYTHONPATH=src
unset ME_ATTENTION_RECORDED_WL
.venv/bin/python -m me_attention.harness --sweep night_replicate --datasets collagen_pepsin,lichens2,lichens1 --models ex_tokens_v2,meas_tokens,smartattention_v0,hybrid_conv_attn --repeats collagen_pepsin=5,lichens2=5,lichens1=1 --budget 3 --max-epochs 25 --device mps --workers 6 --gpu-workers 3 --spatial-cap 2000 --order spectral-first
echo REPLICATE_DONE
