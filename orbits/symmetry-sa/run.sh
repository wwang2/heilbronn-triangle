#!/bin/bash
# Reproduce symmetry-sa optimization
# Run from repo root: bash .worktrees/symmetry-sa/orbits/symmetry-sa/run.sh

set -e

echo "=== Symmetry-aware SA for Heilbronn Triangle (n=11) ==="
echo "Running ultra-fine Numba SA with symmetry enforcement..."

# Run the optimizer (v7: symmetric-enforced + unconstrained refinement)
uv run python .worktrees/symmetry-sa/orbits/symmetry-sa/optimize_v7.py

echo ""
echo "=== Evaluating solution ==="
uv run python research/eval/run_eval.py --evaluate .worktrees/symmetry-sa/orbits/symmetry-sa/solution.py
