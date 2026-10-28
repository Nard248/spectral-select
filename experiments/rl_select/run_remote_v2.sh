#!/usr/bin/env bash
# Full rerun of the reversible-RL band-selection study on a remote machine (docs/rl_select/HANDOFF_REMOTE_RL_v2.md).
# Resumable: every driver skips finished rows/policies, and finished stages leave a marker in $RL_SELECT_RESULTS/.done/.
#
#   WORKERS=15 nohup bash experiments/rl_select/run_remote_v2.sh > /dev/null 2>&1 &
#   tail -f results/rl_select_v2/run.log          # prints RL_V2_DONE at the very end
#
# Env (all optional):  PY=.venv/bin/python  WORKERS=<cpu cores - 1>  SEEDS=0,1,2  FORCE=1 (ignore stage markers)
#                      STAGES="caches baselines rl_s2_s3 rl_s1_abl_s4 rl_v3 report"  (run a subset, in this order)
#                      V3_UPDATES=400 (PPO updates of the augmented policies; code default, see handoff)
set -uo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

PY=${PY:-.venv/bin/python}
if command -v nproc >/dev/null 2>&1; then NCPU=$(nproc); else NCPU=$(sysctl -n hw.ncpu); fi
V3_UPDATES=${V3_UPDATES:-250}  # equal to the non-augmented policies (removes the training-length confound)
WORKERS=${WORKERS:-$(( NCPU > 1 ? NCPU - 1 : 1 ))}
SEEDS=${SEEDS:-0,1,2}
STAGES=${STAGES:-"caches baselines rl_s2_s3 rl_s1_abl_s4 rl_v3 report"}

export PYTHONPATH=src
export RL_SELECT_RESULTS=${RL_SELECT_RESULTS:-results/rl_select_v2}
export RL_SELECT_WORKERS=$WORKERS RL_SELECT_SEEDS=$SEEDS
# one BLAS/OpenMP thread per worker process: CPU-parallel over (source, variant, seed) jobs is the fastest layout
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

RES="$ROOT/$RL_SELECT_RESULTS"; [[ "$RL_SELECT_RESULTS" = /* ]] && RES="$RL_SELECT_RESULTS"
mkdir -p "$RES/logs" "$RES/.done"
LOG="$RES/run.log"
exec > >(tee -a "$LOG") 2>&1

ts() { date '+%Y-%m-%d %H:%M:%S'; }
echo "[$(ts)] RL_V2 start: root=$RES workers=$WORKERS seeds=$SEEDS stages=($STAGES) python=$PY"

FAILED=0
stage() {  # stage <name> <command...>
  local name=$1; shift
  if [[ -f "$RES/.done/$name" && -z "${FORCE:-}" ]]; then echo "[$(ts)] stage $name: already done, skipping"; return 0; fi
  echo "[$(ts)] stage $name: START  ($*)"
  local t0=$SECONDS
  if "$@" > >(tee -a "$RES/logs/$name.log") 2>&1; then
    touch "$RES/.done/$name"; echo "[$(ts)] stage $name: OK in $(( SECONDS - t0 ))s"
  else
    echo "[$(ts)] stage $name: FAILED (exit $?) after $(( SECONDS - t0 ))s -- rerun the script to resume"; FAILED=1
  fi
}

want() { [[ " $STAGES " == *" $1 "* ]]; }

# 1. dataset caches from the ME-attention loader (needs results/me_attention/cache/*_recwl.npz + lichens2_sub3.npz)
caches() { "$PY" experiments/rl_select/build_caches.py; }
# 2. baselines S1/S2/S3 (+ exact n=2,3)
baselines() {
  "$PY" experiments/rl_select/run_baselines.py --phase main --jobs "$WORKERS" &&
  "$PY" experiments/rl_select/run_baselines.py --phase exact --jobs "$WORKERS"
}
# 3-5. RL policy training (6 variants x 6 sources x SEEDS) -> S2 label-free deployment -> S3 few-shot (LODO sources).
#      The driver pipelines them: each finished policy immediately enqueues its S2/S3 deployment jobs.
rl_s2_s3() { "$PY" experiments/rl_select/run_rl.py s2 --workers "$WORKERS" --seeds ${SEEDS//,/ }; }
# 6. per-instance RL search S1, collagen_acetic ablation (paper Table search), hardware-cost S4 (paper Sec. cost)
rl_s1_abl_s4() { "$PY" experiments/rl_select/run_rl.py s1 abl s4 --workers "$WORKERS"; }
# augmented-task policies (aug v3: add, add/remove, CSS-start, mix) x SEEDS, train + S2 deployment
rl_v3() { "$PY" experiments/rl_select/run_rl_v3.py run --workers "$WORKERS" --seeds ${SEEDS//,/ } ${V3_UPDATES:+--updates $V3_UPDATES}; }
# 7. digests, paper tables/figures (-> $RES/paper/), gain decomposition (-> $RES/summary/), count check
report() {
  mkdir -p "$RES/summary" &&
  "$PY" experiments/rl_select/aggregate.py &&
  "$PY" experiments/rl_select/aggregate_rl.py &&
  "$PY" experiments/rl_select/digest_rl_v3.py &&
  "$PY" experiments/rl_select/paper_figures.py &&
  "$PY" experiments/rl_select/decompose_gain.py &&
  "$PY" experiments/rl_select/verify_v2.py
}

for s in caches baselines rl_s2_s3 rl_s1_abl_s4 rl_v3 report; do
  want "$s" && stage "$s" "$s"
done

if [[ $FAILED -eq 0 ]]; then echo "[$(ts)] RL_V2_DONE"; else echo "[$(ts)] RL_V2_INCOMPLETE (a stage failed; see above)"; exit 1; fi
