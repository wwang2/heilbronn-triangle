---
strategy: known-literature
status: complete
eval_version: eval-v1
metric: 0.03630
issue: 4
parents:
  - null
---

## Glossary

- SA: Simulated Annealing
- SOTA: State of the Art (0.03653, AlphaEvolve)
- DE: Differential Evolution
- combined_score: Weighted combination of min_area metric and any penalty terms used by the evaluator; best single run achieved combined_score=0.9937

## Approach

Search for known best configurations from mathematical literature for the Heilbronn
triangle problem with n=11 points in an equilateral triangle, then refine via
numerical optimization.

### Literature findings

- The Heilbronn triangle problem for n=11 in a **unit square** was solved by Goldberg (1972),
  achieving min area = 1/27 ~ 0.0370 with a horizontally symmetric configuration.
- For an **equilateral triangle**, the best known pre-AlphaEvolve value was
  H_11 >= 0.03609267801015405 (from MathWorld, attributed to Cantrell/Friedman).
- AlphaEvolve achieved 0.03653 (the SOTA benchmark in this evaluator).
- Exact coordinates for n=11 configurations are not published in any accessible source.
  The MathWorld, Wikipedia, Friedman packing pages, and arxiv papers (2512.14505, 2603.11107)
  all discuss values but do not publish coordinates for n >= 10.

### Optimization approach

Since exact coordinates are unavailable, we used multi-start simulated annealing:
1. Random initialization inside the equilateral triangle
2. SA with partial area updates (only recompute triplets involving moved point)
3. Focused moves: 40% of moves target points in the current smallest triangle
4. Multi-phase with reheating from best known solution
5. 3M iterations per seed, 20 seeds in parallel

> **Note on solution.py:** `solution.py` is a static lookup — it returns the best coordinates
> found during the SA optimization runs described above. The full SA code (with multi-start,
> reheating, and partial-area updates) ran during exploration and is preserved in the
> `optimize*.py` files. The key output — the hardcoded point coordinates for seed=13 — is
> what `heilbronn_triangle11()` returns directly, so evaluation is instantaneous.

## Results

| Seed | Metric | Time |
|------|--------|------|
| 13   | 0.03630 | 1209s |
| 3    | 0.03553 | 619s |
| 5    | 0.03495 | 621s |
| **Mean top-3** | **0.03559 +/- 0.0007** | |

Best single run: seed=13, metric=0.03630, combined_score=0.9937

Refinement runs (10M iterations, starting from best) are in progress.

## Prior Art & Novelty

### What is already known
- Goldberg (1972) found the n=11 unit square configuration with min area 1/27
- Comellas & Yebra (2002) improved bounds using simulated annealing
- Cantrell/Friedman achieved H_11 >= 0.03609 for equilateral triangles
- AlphaEvolve (DeepMind, 2025) achieved 0.03653 for equilateral triangle n=11

### What this orbit adds
- Independent numerical optimization reaching 99.4% of SOTA
- No novelty claim: this uses standard SA, a well-known optimization technique

### Honest positioning
This orbit applies simulated annealing (a classical optimization method) to a known
problem. The contribution is purely computational - finding good coordinates through
numerical search. The SOTA was set by AlphaEvolve using evolutionary programming with
LLM-guided code mutation, which is a fundamentally different approach.

## References
- [Goldberg (1972)](https://mathworld.wolfram.com/HeilbronnTriangleProblem.html) — original n=11 unit square result
- [Comellas & Yebra (2002)](https://en.wikipedia.org/wiki/Heilbronn_triangle_problem) — improved bounds via SA
- [AlphaEvolve (2025)](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — SOTA 0.03653
- [Sudermann-Merx (2026)](https://arxiv.org/abs/2603.11107) — exact coordinates for n<=9
- [Monji et al. (2025)](https://arxiv.org/abs/2512.14505) — global optimization for n<=10
