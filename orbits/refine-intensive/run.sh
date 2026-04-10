#!/bin/bash
# Reproduce the intensive SA optimization
cd "$(dirname "$0")/../.."
uv run python orbits/refine-intensive/optimize.py
