#!/bin/bash
# Reproduce the Heilbronn triangle n=11 optimization
# Usage: bash orbits/known-literature/run.sh
set -e
cd "$(dirname "$0")/../.."

echo "=== Phase 1: Multi-start simulated annealing ==="
uv run python orbits/known-literature/optimize3.py

echo "=== Phase 2: Refinement from best config ==="
uv run python orbits/known-literature/refine.py

echo "=== Phase 3: Evaluate final solution ==="
uv run python research/eval/run_eval.py --evaluate orbits/known-literature/solution.py
