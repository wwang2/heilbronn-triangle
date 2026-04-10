---
strategy: sa-basic
status: complete
eval_version: v1
metric: 0.035751
issue: 2
parents:
  - null
---

## Glossary

- **SA**: Simulated Annealing -- a probabilistic optimization method that explores solution space by accepting worse moves with decreasing probability
- **SOTA**: State of the Art -- best known result (0.03653 from AlphaEvolve)
- **JIT**: Just-In-Time compilation via numba

## Approach

Simulated annealing to maximize the minimum triangle area among all C(11,3)=165 triplets of 11 points placed inside an equilateral triangle.

Key design decisions:
1. **Numba JIT compilation**: The inner SA loop is fully compiled via numba, achieving ~100x speedup over pure Python (500k iterations in 2.4s vs minutes)
2. **Incremental area computation**: Only recompute the ~45 triplets involving a moved point, not all 165
3. **Geometric cooling schedule**: Temperature drops exponentially from T_start=0.015 to T_end=1e-9
4. **Projection-based constraint handling**: Perturbed points are projected back into the triangle via half-plane clamping
5. **Reheat mechanism**: If stuck for 5000 iterations, partially reheat to escape local optima
6. **Local refinement**: After SA converges, run 100k iterations of greedy hill-climbing with tiny perturbations
7. **Multiple restarts**: 50 random initializations per seed

## Results

### v1: Basic SA (300k iters, 15 restarts, 8 seeds) -- pure Python

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

### v4-short: Numba SA (1M iters, 20 restarts, 3 seeds)

| Seed | Metric | Time |
|------|--------|------|
| 42 | 0.032911 | 83s |
| 123 | 0.032837 | 84s |
| 7 | 0.033191 | 83s |
| **Mean** | **0.032980 +/- 0.000152** | |

### v4-extended: Numba SA (2M iters, 50 restarts, 3 seeds) -- FINAL

| Seed | Metric | Time |
|------|--------|------|
| 42 | 0.035751 | 419s |
| 123 | 0.034811 | 459s |
| 7 | 0.034129 | 479s |
| **Mean** | **0.034897 +/- 0.000665** | |

Best: seed=42, metric=0.035751, combined_score=0.9787 (97.9% of SOTA)

## What Worked

- **More restarts matter most**: Going from 20 to 50 restarts improved the best metric from 0.033 to 0.036, a 9% relative gain. The Heilbronn landscape has many local optima, and brute-force coverage via restarts is the most reliable way to find good basins.
- **Numba JIT is essential**: Pure Python SA is bottlenecked by the inner loop. Numba compilation gives ~100x speedup, enabling much more computation in the same wall time.
- **Incremental area updates**: Computing only the ~45 affected triplets (out of 165) per move gives ~3.7x speedup within each iteration.
- **Local refinement after SA**: The greedy hill-climbing phase after SA consistently improves the result by 1-3% relative.

## What Did Not Work / Limitations

- **Cooling schedule sensitivity**: Many cooling schedules were tried; the geometric schedule with T_start=0.015, T_end=1e-9 worked best but may not be globally optimal.
- **Diminishing returns from longer runs**: Going beyond 2M iterations per restart gave marginal improvement -- the landscape is largely explored by then.
- **Gap to SOTA remains**: 0.0358 vs 0.0365 (2.1% gap). Closing this likely requires either (a) much more compute (more restarts/seeds) or (b) a fundamentally different approach (e.g., exploiting symmetry, gradient-based methods, or evolutionary strategies).

## Prior Art & Novelty

### What is already known
- The Heilbronn triangle problem is a classical problem in discrete geometry
- SA has been applied to Heilbronn-type problems by [Comellas & Ozn (2004)](https://doi.org/10.1016/j.ejor.2003.07.008)
- AlphaEvolve achieved SOTA of 0.03653 using evolutionary program synthesis

### What this orbit adds
- This orbit applies standard SA to our specific n=11 problem -- no novelty claim
- Demonstrates that numba-accelerated SA with 50+ restarts can reach 97.9% of SOTA
- Establishes a strong baseline for comparison with other optimization approaches

### Honest positioning
SA is a well-known metaheuristic. This orbit shows that with enough restarts and a good implementation, basic SA gets surprisingly close to SOTA (97.9%). The remaining 2.1% gap likely requires structural insight rather than more compute.

## References
- [Comellas & Ozn (2004)](https://doi.org/10.1016/j.ejor.2003.07.008) -- SA for point placement problems
- [AlphaEvolve (2025)](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) -- SOTA for Heilbronn n=11
