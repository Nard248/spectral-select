#!/bin/bash
# Phase A of the 2026-10-28 handoff: finish the clean runs that were stopped on the laptop.
#   1. the 8 missing meas_tokens folds of the lichens2_sub3 sweep (completed jobs are skipped automatically)
#   2. the pre-registered ex_tokens_lin variant on lichens2_sub3 and on the recorded-wavelength benchmark
# Usage: DEVICE=cuda WORKERS=8 GPU_WORKERS=2 PY=.venv/bin/python bash experiments/me_attention/run_remote_phaseA.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE=${DEVICE:-cuda}; WORKERS=${WORKERS:-8}; GPU_WORKERS=${GPU_WORKERS:-2}; PY=${PY:-.venv/bin/python}
export PYTHONPATH=src
COMMON="--budget 3 --max-epochs 25 --device $DEVICE --workers $WORKERS --gpu-workers $GPU_WORKERS --order spectral-first"

# lichens2_sub3 has recorded wavelengths built into its config: do NOT set ME_ATTENTION_RECORDED_WL here
# (the variable changes the cache file name to *_recwl.npz, which does not exist for this dataset)
$PY -m me_attention.harness --sweep sub3 --datasets lichens2_sub3 \
    --models lr,rbfsvm,rf,hgb,knn,mlp,cnn1d,spectralformer,smartattention_v0,ex_tokens_v2,hybrid_conv_attn,meas_tokens,ex_tokens_lin \
    --repeats lichens2_sub3=5 --spatial-cap 2000 $COMMON

ME_ATTENTION_RECORDED_WL=1 $PY -m me_attention.harness --sweep recwl --datasets collagen_pepsin,lichens2,lichens1 \
    --models ex_tokens_lin --repeats collagen_pepsin=5,lichens2=5,lichens1=1 $COMMON

# merge the new recwl records into the combined results folder used by the paper analysis
for ds in collagen_pepsin lichens2 lichens1; do
    cp results/me_attention/recwl/$ds/ex_tokens_lin.jsonl results/me_attention/final_recwl/$ds/
done
echo PHASE_A_DONE
