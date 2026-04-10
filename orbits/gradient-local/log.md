---
strategy: gradient-local
status: in-progress
eval_version: v1
metric: 0.036429
issue: 6
parents:
  - orbit/known-literature
---

## Glossary

- NM: Nelder-Mead (derivative-free simplex optimization method)
- SOTA: State of the Art (0.03653, AlphaEvolve)
- LSE: Log-Sum-Exp (smooth approximation to min/max)

## Approach

Gradient-based and derivative-free local optimization starting from the parent
orbit's best configuration (known-literature, metric=0.036301). The core idea
is that the parent config is near a local optimum, and we can refine it by:

1. **Softmin continuation**: Smooth the non-differentiable min-area objective
   using log-sum-exp, then optimize with gradient ascent while gradually
   increasing the sharpness parameter beta.

2. **Targeted Nelder-Mead**: Identify the "bottleneck" triangles (smallest area)
   and optimize only the points participating in those triangles. This reduces
   the search dimension from 22 to ~6-14 variables.

3. **Iterated targeted NM with restarts**: Repeatedly identify bottleneck,
   optimize, update, repeat. When stuck, perturb and restart.

### Key finding

The parent config has remarkably well-equalized triangle areas -- the smallest
5 triangles all have areas within 0.001% of each other. This makes the min-area
landscape very flat locally. The most effective strategy was perturbed restarts
(scale ~5e-3) followed by iterated targeted NM refinement, which found
progressively better basins.

## Results

| Round | Best Method | Best Metric | vs Parent | vs SOTA |
|-------|------------|-------------|-----------|---------|
| R1 | nelder-mead | 0.036302 | +0.004% | -0.63% |
| R2 | targeted-nm | 0.036320 | +0.052% | -0.58% |
| R3 | iter-tnm | 0.036323 | +0.062% | -0.57% |
| R4 | perturbed-restart | 0.036343 | +0.115% | -0.51% |
| R5 | perturbed-restart | 0.036389 | +0.244% | -0.38% |
| R6 | refine+iter-tnm | 0.036429 | +0.354% | -0.28% |

### Round 6 (best so far) detail -- 3+ seeds

| Seed | Scale | Metric | Time |
|------|-------|--------|------|
| 553 | 1e-4 | 0.036429 | 204s |
| 1606 | 1e-4 | 0.036421 | 210s |
| 98 | 1e-4 | 0.036415 | 219s |
| **Mean top-3** | | **0.036422 +/- 0.000007** | |

## Prior Art & Novelty

### What is already known
- Nelder-Mead and other derivative-free methods are standard for non-smooth optimization
- Log-sum-exp softmin for smooth approximation of min is well-known
- Continuation methods (gradually sharpening smooth approximation) are classical

### What this orbit adds
- Application of iterated targeted NM (optimizing only bottleneck-participating
  points) to the Heilbronn problem
- Empirical finding that the known-literature parent config sits near a good
  basin but not the best one -- perturbed restarts at scale ~5e-3 find better
  basins

### Honest positioning
No novelty claim. This orbit applies standard optimization techniques to
a known configuration. The main contribution is improving the parent's metric
from 0.036301 to 0.036429, reducing the gap to SOTA from 0.6% to 0.3%.

## References
- Parent orbit: known-literature (#4), metric=0.036301
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) -- SOTA 0.03653
- [Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
