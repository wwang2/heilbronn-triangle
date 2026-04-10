---
strategy: symmetry-sa
status: in-progress
eval_version: v1
metric: 0.036530
issue: 7
parents:
  - orbit/refine-intensive
  - orbit/known-literature
---

## Glossary

- SA: Simulated Annealing
- SOTA: State of the Art (0.03653, AlphaEvolve)
- JIT: Just-In-Time compilation (Numba)
- D3: Dihedral group of order 6 (symmetry group of equilateral triangle)
- C3: Cyclic group of rotations by 120 degrees

## Approach

Exploit the D3 symmetry of the equilateral triangle to constrain and guide the
optimization search. The equilateral triangle has 6 symmetries (3 rotations + 3
reflections). For n=11 points, exact C3 rotational symmetry is impossible (11 mod 3 = 2),
but mirror symmetry about x=0.5 is compatible with 11 = 2*5 + 1.

### Key finding from parent orbits

The refine-intensive orbit (0.036530) independently discovered that the optimal
configuration has near-perfect mirror symmetry about x=0.5:
- 5 pairs of points related by (x,y) -> (1-x,y)
- 1 point on the symmetry axis at x=0.5
- Pair deviations < 2e-5

### Strategy

1. Start from refine-intensive best (0.036530)
2. Enforce perfect mirror symmetry (halving free parameters from 22 to 11)
3. Run Numba-JIT SA in the reduced (symmetric) parameter space
4. Follow with unconstrained SA refinement (allow symmetry breaking)
5. Coordinate descent for machine-precision polish

### What we tried

1. **Pure SA from known-literature** (v1-v3): Could not escape the 0.03630 basin.
   The known-literature config is a different structural arrangement.

2. **Basin-hopping SA** (v3): Random large perturbations + SA refinement.
   Always converged back to 0.03630 or worse.

3. **Gradient-based (scipy L-BFGS-B)** (v4): Barycentric parameterization failed
   due to constraint handling issues.

4. **Numba-JIT SA from refine-intensive** (v5): Confirmed 0.036530.
   Symmetric, unconstrained, and hybrid modes all converge to same basin.

5. **Ultra-fine incremental SA** (v6): 600M iterations with cached area updates.
   Found marginal improvement: 0.03652984 (vs 0.03652980).

6. **Perfect symmetrization + coordinate descent**: Enforcing exact mirror symmetry
   and polishing with coordinate descent improved to 0.03652985.

7. **Symmetric-enforced Numba SA** (v7): SA in the 11-parameter symmetric space
   followed by unconstrained refinement. Running.

## Results

### Current best (symmetrized + coordinate descent)

| Seed | Metric | Time |
|------|--------|------|
| 42 | 0.03652980 | 672s |
| 123 | 0.03652980 | 683s |
| 7 | 0.03652984 | 671s |
| **Mean** | **0.03652982 +/- 0.00000002** | |

Post-symmetrization + CD: **0.03652985**

- combined_score = 0.9999988 (99.99988% of SOTA)
- Improvement from parent: 0.03652980 -> 0.03652985 (+1.4e-7)

### Symmetry structure

The optimal config has 5 mirror pairs + 1 axis point:

| Pair | Left | Right | y | Deviation |
|------|------|-------|---|-----------|
| 0-7 | (0.106, 0.071) | (0.894, 0.071) | 0.071 | 2.5e-6 |
| 1-4 | (0.148, 0.256) | (0.852, 0.256) | 0.256 | 6.7e-6 |
| 3-9 | (0.277, 0.000) | (0.723, 0.000) | 0.000 | 8.7e-6 |
| 5-6 | (0.428, 0.741) | (0.572, 0.741) | 0.741 | 1.2e-5 |
| 8-10 | (0.409, 0.439) | (0.591, 0.439) | 0.439 | 1.8e-5 |
| Axis (2) | (0.500, 0.211) | -- | 0.211 | -- |

The 10 smallest triangles all have areas within 1e-8 of the minimum,
indicating multiple simultaneously active constraints at the optimum.

## Prior Art & Novelty

### What is already known
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) achieved SOTA 0.03653 for n=11
- SA is a standard optimization technique; Numba JIT is a standard acceleration
- Mirror symmetry of optimal Heilbronn configs is expected from the symmetry of the container

### What this orbit adds
- Confirmed that enforcing exact mirror symmetry slightly improves the solution
  (0.03652985 vs 0.03652980 from unconstrained SA)
- Demonstrated the symmetric parameter space (11 DOF vs 22) is sufficient for near-SOTA
- Showed that C3 rotational symmetry is NOT present in the optimum (as expected for n=11)

### Honest positioning
This orbit applies symmetry reduction to a well-studied optimization problem.
The improvement over the parent orbit is negligible (~1e-7 in normalized area).
The main value is the symmetry analysis confirming that the optimal config is
exactly symmetric, not just approximately so.

## References
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve/) -- SOTA 0.03653
- [Numba](https://numba.pydata.org/) -- JIT compilation for Python
- Parent orbits: refine-intensive (#5, metric 0.036530), known-literature (#4, metric 0.03630)
