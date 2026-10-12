#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}
"$PYTHON" "$ROOT/export_dataset.py"
mkdir -p "$ROOT/build" "$ROOT/results"
clang++ -std=c++20 -O3 -ffast-math -mcpu=native -pthread "$ROOT/src/main.cpp" -o "$ROOT/build/native_mc_rl"
"$ROOT/build/native_mc_rl" "$ROOT/data/lichens2.bin" "$ROOT/results" 13 30 3
