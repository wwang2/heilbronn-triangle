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
