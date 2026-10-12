#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON=${PYTHON:-python3}

"$PYTHON" "$SCRIPT_DIR/export_dataset.py"
mkdir -p "$SCRIPT_DIR/build"
clang++ -std=c++20 -O3 -ffast-math -mcpu=native -pthread \
  "$SCRIPT_DIR/src/main.cpp" -o "$SCRIPT_DIR/build/lichen_exact"
mkdir -p "$SCRIPT_DIR/results"
"$SCRIPT_DIR/build/lichen_exact" "$SCRIPT_DIR/data/lichens2.bin" "$SCRIPT_DIR/results/top_13.csv" 13
