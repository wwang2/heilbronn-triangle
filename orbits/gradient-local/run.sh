#!/bin/bash
# Reproduce gradient-local optimization
# Phase 1: Targeted NM (optional, SA is the main contributor)
# Phase 2: SA refinement
cd "$(dirname "$0")/../.."
uv run python orbits/gradient-local/optimize_sa.py
uv run python orbits/gradient-local/optimize_sa2.py
uv run python orbits/gradient-local/optimize_sa3.py
uv run python research/eval/run_eval.py --evaluate orbits/gradient-local/solution.py
