#!/bin/bash
set -e
cd "$(dirname "$0")/../.."

echo "=== Running SA optimization ==="
uv run python orbits/sa-basic/optimize.py

echo ""
echo "=== Evaluating solution ==="
uv run python research/eval/run_eval.py --evaluate orbits/sa-basic/solution.py
