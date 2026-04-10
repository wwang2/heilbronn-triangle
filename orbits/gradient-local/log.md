---
strategy: gradient-local
status: complete
eval_version: v1
metric: 0.036530
issue: 6
parents:
  - orbit/known-literature
---

## Glossary

- NM: Nelder-Mead (derivative-free simplex optimization method)
- SA: Simulated Annealing
- SOTA: State of the Art (0.03653, AlphaEvolve)
- LSE: Log-Sum-Exp (smooth approximation to min/max)

## Approach

Local optimization starting from the parent orbit's best configuration
(known-literature, metric=0.036301). Two phases:

### Phase 1: Targeted Nelder-Mead (rounds R1-R6)

The parent config has remarkably well-equalized triangle areas -- the smallest
5 triangles all have areas within 0.001% of each other. This makes the min-area
landscape very flat locally.

The key idea was **targeted NM**: identify the "bottleneck" triangles (smallest
area), extract only the points participating in those triangles (reducing the
search space from 22 to ~6-14 variables), then run NM on that subspace. After
convergence, re-identify the new bottleneck and repeat.

Perturbed restarts at scale ~5e-3 proved essential for escaping local basins.
This phase improved metric from 0.036301 to 0.036429 (99.7% of SOTA).

### Phase 2: Simulated Annealing (rounds SA1-SA3)

SA was dramatically more effective than NM-based methods. Starting from the
NM-refined configuration, SA with targeted moves (50-60% of moves target
bottleneck-triangle points) and partial area updates (recompute only the ~45
triplets involving the moved point) achieved near-SOTA results.

Key SA parameters for the final round:
- Temperature: T0=2e-6, T_final=1e-11 (very cold -- we start near optimum)
- Steps: 10M per seed
- Step size: 3e-4 with adaptive scaling
- Bottleneck targeting: 50% of moves

## Results

| Round | Method | Best Metric | vs SOTA | Combined Score |
|-------|--------|-------------|---------|----------------|
| Parent | -- | 0.036301 | -0.63% | 0.9937 |
| R1 | NM | 0.036302 | -0.63% | 0.9938 |
| R2 | Targeted NM | 0.036320 | -0.58% | 0.9942 |
| R3 | Iterated targeted NM | 0.036323 | -0.57% | 0.9943 |
| R4 | Perturbed restart + NM | 0.036343 | -0.51% | 0.9949 |
| R5 | Perturbed restart + NM | 0.036389 | -0.38% | 0.9962 |
| R6 | Refine + iterated NM | 0.036429 | -0.28% | 0.9972 |
| SA1 | Simulated annealing | 0.036528 | -0.006% | 0.9999 |
| SA2 | Refined SA | 0.036530 | -0.0007% | 0.99999 |
| **SA3** | **Final SA** | **0.036530** | **-0.0003%** | **0.999997** |

### Final result (SA3) -- 3+ seeds

| Seed | Metric | Gap to SOTA | Time |
|------|--------|-------------|------|
| 383 | 0.0365297943 | -9.56e-8 | 300s |
| 321 | 0.0365297927 | -9.72e-8 | 300s |
| 290 | 0.0365297701 | -1.20e-7 | 300s |
| **Mean top-3** | **0.036530 +/- 0.000000013** | | |

## Prior Art & Novelty

### What is already known
- Simulated annealing for Heilbronn-type problems: Comellas and Yebra (2002)
- Nelder-Mead is standard for non-smooth optimization
- Partial area updates for SA on point placement: common in computational geometry

### What this orbit adds
- Empirical finding that targeted moves (focusing on bottleneck-triangle points)
  dramatically improve SA convergence for this problem
- Two-phase approach: NM to reach a good basin, then SA to refine to near-SOTA
- Demonstration that the parent's SA-found config (0.03630) was not in the
  globally best basin -- perturbations + NM + SA found a 0.63% better basin

### Honest positioning
No novelty claim. This orbit applies standard optimization techniques
(Nelder-Mead, simulated annealing) to a known configuration. The main
contribution is reaching 99.9997% of SOTA from a 99.37% starting point,
demonstrating the effectiveness of targeted SA with partial updates.

## References
- Parent orbit: known-literature (#4), metric=0.036301
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) -- SOTA 0.03653
- [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
- [Simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing)
- [Comellas & Yebra (2002)](https://doi.org/10.1016/S0925-7721(01)00055-7) -- SA for Heilbronn
