#!/bin/bash
# Reproduce gradient-local optimization
cd "$(dirname "$0")/../.."
uv run python orbits/gradient-local/optimize6.py
uv run python research/eval/run_eval.py --evaluate orbits/gradient-local/solution.py
