---
strategy: sa-basic
status: in-progress
eval_version: v1
metric: 0.032625
issue: 2
parents:
  - null
---

## Glossary

- **SA**: Simulated Annealing — a probabilistic optimization method that explores solution space by accepting worse moves with decreasing probability
- **SOTA**: State of the Art — best known result (0.03653 from AlphaEvolve)

## Approach

Simulated annealing to maximize the minimum triangle area among all C(11,3)=165 triplets of 11 points placed inside an equilateral triangle.

Key design decisions:
1. **Incremental area computation**: Only recompute the ~45 triplets involving a moved point, not all 165
2. **Geometric cooling schedule**: Temperature drops exponentially from T_start to T_end
3. **Projection-based constraint handling**: Perturbed points are projected back into the triangle
4. **Reheat mechanism**: If stuck for 5000 iterations, partially reheat to escape local optima
5. **Local refinement**: After SA converges, run greedy hill-climbing with tiny perturbations
6. **Multiple restarts**: 20-30 random initializations per seed

## Results

### v1: Basic SA (300k iters, 15 restarts, 8 seeds)

| Seed | Metric | Time |
|------|--------|------|
| 42 | 0.030133 | 444s |
| 123 | 0.031181 | 444s |
| 7 | 0.029327 | 445s |
| 999 | 0.031422 | 444s |
| 2024 | 0.032455 | 447s |
| 314 | 0.031017 | 447s |
| 271 | 0.030008 | 443s |
| 1618 | 0.032625 | 448s |
| **Mean** | **0.031021 +/- 0.001090** | |

Best: seed=1618, metric=0.032625, combined_score=0.8931

### v3: Fast SA with incremental updates (in progress)
- 1M iterations, 30 restarts, 100k refinement steps
- Running with seeds [42, 123, 7]

## Prior Art & Novelty

### What is already known
- The Heilbronn triangle problem is a classical problem in discrete geometry
- [Comellas & Ozn (2004)](https://doi.org/10.1016/j.ejor.2003.07.008) applied SA to Heilbronn-type problems
- AlphaEvolve achieved SOTA of 0.03653 using evolutionary program synthesis

### What this orbit adds
- This orbit applies standard SA to our specific problem — no novelty claim
- Serves as a baseline for more sophisticated optimization approaches

### Honest positioning
SA is a well-known metaheuristic. This orbit establishes a baseline performance level that other approaches (genetic algorithms, gradient-based methods, hybrid approaches) can be compared against.

## References
- [Comellas & Ozn (2004)](https://doi.org/10.1016/j.ejor.2003.07.008) — SA for point placement problems
- [AlphaEvolve (2025)](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — SOTA for Heilbronn n=11
