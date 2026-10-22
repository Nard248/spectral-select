#!/bin/bash
# Combine 19 Oct baselines ('night') with the recorded-wavelength attention rerun ('recwl') into sweep 'final_recwl',
# regenerate tables/figures/numbers into publications/me_attention/generated_recwl, and rerun the importance analysis.
set -e
cd /Users/narekmeloyan/PycharmProjects/4D-Hyperspectral-Unsupervised-Clustering
export PYTHONPATH=src PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2
R=results/me_attention
rm -rf $R/final_recwl
for ds in collagen_pepsin lichens1 lichens2; do
  mkdir -p $R/final_recwl/$ds
  for f in $R/night/$ds/*.jsonl; do
    m=$(basename $f .jsonl)
    if [ -f $R/recwl/$ds/$m.jsonl ]; then cp $R/recwl/$ds/$m.jsonl $R/final_recwl/$ds/; else cp $f $R/final_recwl/$ds/; fi
  done
done
echo "final_recwl = night baselines + recwl attention models (built $(date))" > $R/final_recwl/README.txt
OUT=publications/me_attention/generated_recwl
mkdir -p $OUT
.venv/bin/python -m me_attention.analysis --sweep final_recwl --out $OUT
cp -r publications/me_attention/generated/datasets $OUT/ 2>/dev/null || true
for ds in collagen_pepsin lichens1 lichens2; do
  for m in ex_tokens_v2 meas_tokens; do
    ME_ATTENTION_RECORDED_WL=1 nice -n 5 .venv/bin/python experiments/me_attention/importance.py --dataset $ds --model $m --sweep recwl --repeat 0 --fold 0 --device mps --max-pixels 20000 --out $OUT/importance/${ds}_${m} || echo "FAILED importance $ds $m"
  done
done
echo FINALIZE_DONE
