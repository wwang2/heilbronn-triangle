# Steering Log

## Autorun started
- Problem: Heilbronn triangle n=11 — maximize minimum triangle area among 165 triplets
- Metric: Normalized min area (maximize)
- SOTA: 0.03653 (AlphaEvolve)
- Evaluator: Fetched from OpenEvolve repo (algorithmicsuperintelligence/openevolve)
- Review mode: strict (autorun overrides active)
- Max orbits: 30
- GitHub: https://github.com/wwang2/heilbronn-triangle
- Campaign Issue: #1

## Ultra-research loop started
- [DISPATCH MODE] native — all agents via Claude Code
- Orbits spawned: sa-basic (#2), scipy-multistart (#3), known-literature (#4)
- All 3 agents running in background
- Next: monitor for completion, then verify + completion-review

## Iteration 1: Initial exploration batch
- EXPLORE sa-basic: Simulated annealing with random restarts
- EXPLORE scipy-multistart: Scipy basin-hopping with softmin objective  
- EXPLORE known-literature: Literature search for known best configs

## Milestone 1 (3 orbits complete)
- **Best:** known-literature 0.036301 (99.4% SOTA)
- sa-basic: 0.035751 (97.9%) — RETRACTED (code-claims mismatch, coords valid)
- scipy-multistart: 0.03474 (95.1%) — promising, cleaned up
- Gap to SOTA: 0.000229 (0.6%)
- Strategy: next orbits should focus on closing the last 0.6% gap via:
  - REFINE known-literature (more intensive local optimization)
  - EXTEND with hybrid gradient + SA approaches
  - EXPLORE symmetry exploitation or mathematical insight

## Iteration 2: Refinement batch (closing 0.6% gap)
- [ULTRA-RESEARCH] auto-accepted milestone 1 action mix
- CONCLUDED: sa-basic (dead-end, RETRACT), scipy-multistart (superseded)
- EXTEND known-literature → refine-intensive (#5): Numba SA, 100+ restarts, 5M+ iters
- EXTEND known-literature → gradient-local (#6): L-BFGS-B + softmin continuation method
- EXPLORE symmetry-sa (#7): D3 symmetry-constrained search
- All 3 agents dispatched in background
