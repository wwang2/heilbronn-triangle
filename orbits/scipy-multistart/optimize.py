"""
Heavy optimization script for Heilbronn Triangle n=11.
Runs basin-hopping with multiple seeds in parallel, finds best configuration.
"""

import numpy as np
from itertools import combinations
from multiprocessing import Pool
import time

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
EQUIL_AREA = np.sqrt(3) / 4
BENCHMARK = 0.036529889880030156

TRIPLETS = np.array(list(combinations(range(11), 3)))


def project_vec(points):
    """Vectorized projection into equilateral triangle."""
    h = np.sqrt(3) / 2
    l2 = points[:, 1] / h
    l1 = points[:, 0] - 0.5 * l2
    l0 = 1.0 - l1 - l2
    l0 = np.maximum(l0, 0.0)
    l1 = np.maximum(l1, 0.0)
    l2 = np.maximum(l2, 0.0)
    s = l0 + l1 + l2
    s = np.where(s > 0, s, 1.0)
    l0 /= s; l1 /= s; l2 /= s
    return np.column_stack([l1 + 0.5 * l2, l2 * h])


def triangle_areas(points):
    """Compute all 165 triangle areas."""
    p1 = points[TRIPLETS[:, 0]]
    p2 = points[TRIPLETS[:, 1]]
    p3 = points[TRIPLETS[:, 2]]
    return 0.5 * np.abs(
        (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
        - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    )


def softmin_lse(areas, beta):
    scaled = -beta * areas
    mx = np.max(scaled)
    return -(1.0 / beta) * (mx + np.log(np.sum(np.exp(scaled - mx))))


def objective(x, beta):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -softmin_lse(areas, beta)


class TakeStep:
    def __init__(self, stepsize, rng):
        self.stepsize = stepsize
        self.rng = rng
    def __call__(self, x):
        x += self.rng.uniform(-self.stepsize, self.stepsize, size=x.shape)
        points = project_vec(x.reshape(11, 2))
        x[:] = points.flatten()
        return x


def random_tri_points(n, rng):
    points = np.empty((n, 2))
    for i in range(n):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        points[i] = V0 + r1 * (V1 - V0) + r2 * (V2 - V0)
    return points


def run_one_seed(seed):
    """Run full optimization for one seed."""
    from scipy.optimize import basinhopping
    
    t0 = time.time()
    rng = np.random.RandomState(seed)
    
    best_min_area = -np.inf
    best_points = None
    
    n_starts = 8
    
    for start_idx in range(n_starts):
        x0 = random_tri_points(11, rng).flatten()
        current_x = x0.copy()
        
        # Progressive beta schedule
        schedule = [
            (20, 30, 0.12),
            (80, 25, 0.08),
            (200, 25, 0.05),
            (500, 20, 0.03),
            (1500, 15, 0.015),
        ]
        
        for beta, n_iter, stepsize in schedule:
            take_step = TakeStep(stepsize, rng)
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "args": (beta,),
                "options": {"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10},
            }
            result = basinhopping(
                objective, current_x,
                minimizer_kwargs=minimizer_kwargs,
                niter=n_iter,
                take_step=take_step,
                seed=int(rng.randint(0, 2**31)),
                disp=False,
            )
            current_x = result.x.copy()
        
        # Evaluate true min area
        final_points = project_vec(current_x.reshape(11, 2))
        areas = triangle_areas(final_points)
        min_area = np.min(areas)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = final_points.copy()
    
    elapsed = time.time() - t0
    normalized = best_min_area / EQUIL_AREA
    return {
        "seed": seed,
        "min_area": best_min_area,
        "normalized": normalized,
        "combined_score": normalized / BENCHMARK,
        "points": best_points,
        "time": elapsed,
    }


if __name__ == "__main__":
    seeds = [42, 123, 7]
    
    print(f"Running optimization with {len(seeds)} seeds in parallel...")
    print(f"SOTA benchmark: {BENCHMARK}")
    print()
    
    with Pool(len(seeds)) as p:
        results = p.map(run_one_seed, seeds)
    
    # Print results table
    print(f"{'Seed':>6} | {'Normalized':>12} | {'Score':>8} | {'Time':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:>6} | {r['normalized']:>12.8f} | {r['combined_score']:>8.4f} | {r['time']:>7.1f}s")
    
    # Mean and std
    norms = [r['normalized'] for r in results]
    mean_n = np.mean(norms)
    std_n = np.std(norms)
    print("-" * 50)
    print(f"{'Mean':>6} | {mean_n:>12.8f} +/- {std_n:.8f}")
    print(f"{'Score':>6} | {mean_n / BENCHMARK:>12.6f}")
    
    # Find best overall
    best = max(results, key=lambda r: r['normalized'])
    print(f"\nBest seed: {best['seed']}, normalized: {best['normalized']:.8f}")
    print(f"Best points:\n{repr(best['points'])}")
    
    # Output for hardcoding
    print("\n# For hardcoding in solution.py:")
    print("BEST_POINTS = np.array([")
    for p in best['points']:
        print(f"    [{p[0]:.15f}, {p[1]:.15f}],")
    print("])")
