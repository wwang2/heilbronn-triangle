#!/bin/bash
# Reproduce the Heilbronn triangle optimization
# Runs basin-hopping + DE + local search with 3 seeds in parallel
set -e
cd "$(dirname "$0")/../.."
uv pip install scipy
uv run python orbits/scipy-multistart/optimize_v4.py
