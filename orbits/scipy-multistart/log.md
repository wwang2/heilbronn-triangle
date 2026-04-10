---
strategy: scipy-multistart
status: in-progress
eval_version: 1
metric: 0.03474
issue: 3
parents:
  - null
---

## Glossary

- **DE**: Differential Evolution (scipy global optimizer)
- **L-BFGS-B**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Box constraints
- **LSE**: Log-Sum-Exp (smooth approximation of min/max)
- **SOTA**: State of the Art

## Approach

Maximize the minimum triangle area among all C(11,3)=165 triplets of 11 points in an equilateral triangle, using scipy optimization.

### Core Idea

The min over 165 triangle areas is non-smooth, making gradient-based optimization difficult. We approximate it with a log-sum-exp softmin:

    softmin(areas) = -1/beta * log(sum(exp(-beta * area_i)))

For large beta, this closely approximates the true minimum while remaining differentiable. We use L-BFGS-B (which exploits gradients) as the local optimizer within scipy's basin-hopping framework for global search.

### Key Design Decisions

1. **Progressive beta schedule**: Start with small beta (smooth landscape, easy to explore) and progressively increase beta (sharper approximation of true min). This is a form of continuation method.

2. **Projection-based constraints**: Points are projected back into the triangle after each step using barycentric coordinate clipping. This is simpler than constrained optimization and works well with basin-hopping.

3. **Multi-strategy optimization** (V4):
   - Differential Evolution for global search
   - Basin-hopping with L-BFGS-B for local refinement
   - Nelder-Mead polishing on true (non-smooth) min objective
   - Intensive local perturbation around best known solution

4. **Multiple perturbation strategies**: uniform, Gaussian, single-point, centroid-directed moves.

## Results

### V1: Basic basin-hopping (8 starts, 4-stage beta)

| Seed | Normalized | Time |
|------|-----------|------|
| 42   | 0.03094   | 74s  |
| 123  | 0.03452   | 69s  |
| 7    | 0.03287   | 69s  |
| **Mean** | **0.03277 +/- 0.00146** | |

### V2: More starts (8), more beta stages, Nelder-Mead polish

| Seed | Normalized | Time |
|------|-----------|------|
| 42   | 0.03350   | 267s |
| 123  | 0.03294   | 247s |
| 7    | 0.03464   | 253s |
| **Mean** | **0.03369 +/- 0.00071** | |

### V3: Even more starts (15) — diminishing returns

| Seed | Normalized | Time |
|------|-----------|------|
| 42   | 0.03282   | 326s |
| 123  | 0.03289   | 314s |
| 7    | 0.03354   | 321s |
| **Mean** | **0.03309 +/- 0.00033** | |

### V4: DE + basin-hopping + local search around best known

| Seed | Normalized | Score | Time |
|------|-----------|-------|------|
| 42   | 0.03472   | 0.9505 | 254s |
| 123  | 0.03474   | 0.9511 | 253s |
| 7    | 0.03471   | 0.9502 | 247s |
| **Mean** | **0.03472 +/- 0.00001** | **0.9506** | |

Best: 0.03474 (seed 123) = 95.1% of SOTA (0.03653)

## What Happened

1. V1-V2: Basin-hopping with softmin works but is inconsistent across seeds (high variance).
2. V3: More restarts did not help — spreading compute thin across starts is worse than fewer, deeper searches.
3. V4: Combining DE for global search with intensive local perturbation around the best-known solution dramatically reduced variance and slightly improved the best result. The local search converges to a stable basin near the optimum.

## What I Learned

- The Heilbronn landscape has many local optima. Basin-hopping alone finds different basins each run.
- Local search around a good known solution is more effective than blind global search once you have a reasonable starting point.
- The softmin LSE approximation with progressive beta schedule is a solid continuation method for this class of non-smooth optimization problems.
- The gap to SOTA (0.03653) suggests our best solution is in a suboptimal basin. Reaching SOTA likely requires either a qualitatively different search strategy or much more compute.

## Prior Art & Novelty

### What is already known
- The Heilbronn triangle problem is classical; n=11 results were studied by [Yang et al. (2002)](https://link.springer.com/article/10.1007/s00454-002-2776-6).
- Basin-hopping was introduced by [Wales & Doye (1997)](https://doi.org/10.1021/jp970984n) for molecular energy minimization.
- Log-sum-exp smoothing of min/max is a standard technique in optimization.
- The SOTA of 0.03653 was found by AlphaEvolve (Google DeepMind, 2025).

### What this orbit adds
- This orbit applies known optimization techniques (basin-hopping, LSE softmin, DE) to the Heilbronn problem. No novelty claim.
- The progressive beta schedule (continuation method) combined with multi-strategy local search is a practical recipe that others may find useful.

### Honest positioning
This is a straightforward application of existing global optimization methods to a known problem. The result (95% of SOTA) demonstrates that scipy-based optimization can get close but not match evolutionary/RL-based approaches like AlphaEvolve.

## References
- [Wales & Doye (1997)](https://doi.org/10.1021/jp970984n) — basin-hopping for global optimization
- [AlphaEvolve (2025)](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — SOTA for Heilbronn n=11
