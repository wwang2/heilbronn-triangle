---
strategy: refine-intensive
status: complete
eval_version: v1
metric: 0.036530
issue: 5
parents:
  - orbit/known-literature
---

## Glossary

- SA: Simulated Annealing
- SOTA: State of the Art (0.03653, AlphaEvolve)
- JIT: Just-In-Time compilation (Numba)

## Approach

Intensive SA refinement of the parent orbit's best configuration (0.036301, 99.4% of SOTA),
using Numba JIT compilation for a ~100x speedup in the inner SA loop.

The key insight is that near the optimum, improvements come from very fine perturbations
(scale 0.001-0.0001) over hundreds of millions of iterations. With Numba achieving 2.6M
iterations/second, we can afford 800M iterations per seed while staying under 10 minutes
wall-clock time.

### Design choices

1. **Numba JIT inner loop**: The SA accept/reject loop, triangle area computation, and
   incremental area updates are all compiled with `@njit`. This gives ~2.6M iters/sec
   vs ~25K iters/sec in pure Python.

2. **Incremental area updates**: When moving point i, only recompute the 45 triplets
   involving that point (out of 165 total). This is a 3.7x speedup per iteration.

3. **Multi-phase cooling with reheating**: Each phase starts from the global best and
   runs with progressively finer perturbations and lower temperatures. This allows
   escaping local optima in early phases while polishing in later phases.

4. **Focused moves**: With probability 0.35-0.55 (increasing per phase), the moved
   point is chosen from the current minimum-area triangle. This focuses search effort
   where it matters most.

### Round 1: 6 phases x 50M iterations = 300M per seed

Starting from the parent config (0.036301), all 3 seeds found the same structural
basin and converged to metric ~0.036529.

### Round 2: 8 phases x 100M iterations = 800M per seed

Starting from round 1's best (0.036529), ultra-fine perturbations (down to 0.00001)
squeezed out another 0.0000007, reaching 0.036530.

## Results

### Round 1 (300M iterations per seed)

| Seed | Metric | Time |
|------|--------|------|
| 123 | 0.0365291 | 184s |
| 42 | 0.0365284 | 186s |
| 7 | 0.0365215 | 186s |
| **Mean** | **0.036526 +/- 0.000003** | |

### Round 2 (800M iterations per seed, from round 1 best)

| Seed | Metric | Time |
|------|--------|------|
| 42 | 0.0365298 | 489s |
| 123 | 0.0365296 | 490s |
| 7 | 0.0365295 | 490s |
| **Mean** | **0.0365296 +/- 0.0000001** | |

### Final evaluation

- metric = 0.036530 (normalized min area)
- combined_score = 0.999998 (99.9998% of SOTA)
- Improvement from parent: 0.036301 -> 0.036530 (+0.63%)

### Symmetry observations

The optimal configuration exhibits near-perfect mirror symmetry about x = 0.5, with
all point-to-reflected-point distances < 0.00002. The 5 smallest triangles have nearly
equal areas (spread < 1e-8), which is characteristic of a true optimum where multiple
constraints are simultaneously active.

## Prior Art & Novelty

### What is already known
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) achieved SOTA 0.03653 for n=11
- SA is a standard optimization technique; Numba JIT is a standard acceleration technique
- The optimal configuration structure (symmetric, multiple tied minimum triangles) is expected from optimization theory

### What this orbit adds
- Independently found a configuration matching SOTA to 99.9998%
- Confirmed the optimal structure has mirror symmetry about x=0.5
- Demonstrated that Numba-JIT SA can achieve near-SOTA in ~8 minutes on a laptop

### Honest positioning
This orbit applies well-known techniques (SA + JIT compilation) to find near-optimal
coordinates. No algorithmic novelty is claimed. The contribution is purely computational --
demonstrating that intensive local search from a good starting point can essentially
match AlphaEvolve's result for this problem size.

## References
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) -- SOTA 0.03653
- [Numba](https://numba.pydata.org/) -- JIT compilation for Python
- Parent orbit: known-literature (#4) -- starting configuration 0.036301
