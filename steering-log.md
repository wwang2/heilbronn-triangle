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

## Milestone 2 (6 orbits complete, SOTA matched)
- **Best:** symmetry-sa 0.036529884 (combined_score=0.9999998)
- refine-intensive: 0.036529802 (99.9998%) — READY
- gradient-local: 0.036529794 (99.9997%) — under review
- known-literature: 0.036301 (99.4%) — parent, READY
- scipy-multistart: 0.03474 (95.1%) — concluded
- sa-basic: 0.035751 (97.9%) — dead-end (RETRACT)
- **Key insight:** All 3 batch-2 orbits independently converged to the same basin
- **Symmetry discovery:** Optimal n=11 config has exact mirror symmetry about x=0.5
- **Gap to SOTA benchmark:** ~6e-9 (at numerical precision limits)
- **Decision:** SOTA is effectively matched. Consider concluding campaign.

## Campaign Concluded
- **Termination reason:** Target met — SOTA effectively matched (gap < 1e-8)
- **Total orbits:** 7 (4 explore, 2 extend, 1 dead-end)
- **Best metric:** 0.036529884 from symmetry-sa (99.99998% of SOTA)
- **Key finding:** Optimal n=11 config has exact mirror symmetry about x=0.5
- **Search efficiency:** 7 orbits, 2 batches, no anti-stuck brainstorms needed
