"""
V2: More aggressive optimization combining differential evolution + basin-hopping.
Also tries polishing the best solution found with Nelder-Mead on true min objective.
"""

import numpy as np
from itertools import combinations
from multiprocessing import Pool
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
EQUIL_AREA = np.sqrt(3) / 4
BENCHMARK = 0.036529889880030156
TRIPLETS = np.array(list(combinations(range(11), 3)))


def project_vec(points):
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


def objective_smooth(x, beta):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -softmin_lse(areas, beta)


def objective_true(x):
    """True min-area objective (non-smooth)."""
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -np.min(areas)


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


def polish_with_nelder_mead(points, n_iter=500):
    """Polish solution with Nelder-Mead on true min objective."""
    from scipy.optimize import minimize
    x0 = points.flatten()
    result = minimize(
        objective_true, x0,
        method='Nelder-Mead',
        options={'maxiter': n_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True},
    )
    final = project_vec(result.x.reshape(11, 2))
    return final


def run_one_seed(seed):
    """Run full optimization for one seed."""
    from scipy.optimize import basinhopping, differential_evolution
    
    t0 = time.time()
    rng = np.random.RandomState(seed)
    
    best_min_area = -np.inf
    best_points = None
    
    n_starts = 12
    
    for start_idx in range(n_starts):
        x0 = random_tri_points(11, rng).flatten()
        current_x = x0.copy()
        
        # Progressive beta schedule with more iterations
        schedule = [
            (15, 40, 0.15),
            (50, 30, 0.10),
            (150, 25, 0.06),
            (400, 20, 0.04),
            (1000, 15, 0.02),
            (3000, 10, 0.01),
        ]
        
        for beta, n_iter, stepsize in schedule:
            take_step = TakeStep(stepsize, rng)
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "args": (beta,),
                "options": {"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
            }
            result = basinhopping(
                objective_smooth, current_x,
                minimizer_kwargs=minimizer_kwargs,
                niter=n_iter,
                take_step=take_step,
                seed=int(rng.randint(0, 2**31)),
                disp=False,
            )
            current_x = result.x.copy()
        
        # Polish with Nelder-Mead on true objective
        final_points = project_vec(current_x.reshape(11, 2))
        final_points = polish_with_nelder_mead(final_points, n_iter=2000)
        
        areas = triangle_areas(final_points)
        min_area = np.min(areas)
        
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = final_points.copy()
    
    # Extra polish: perturb best solution many times
    for _ in range(20):
        perturbed = best_points + rng.uniform(-0.02, 0.02, size=best_points.shape)
        perturbed = project_vec(perturbed)
        polished = polish_with_nelder_mead(perturbed, n_iter=3000)
        areas = triangle_areas(polished)
        min_area = np.min(areas)
        if min_area > best_min_area:
            best_min_area = min_area
            best_points = polished.copy()
    
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
    
    print(f"Running V2 optimization with {len(seeds)} seeds in parallel...")
    print(f"SOTA benchmark: {BENCHMARK}")
    print()
    
    with Pool(len(seeds)) as p:
        results = p.map(run_one_seed, seeds)
    
    print(f"{'Seed':>6} | {'Normalized':>12} | {'Score':>8} | {'Time':>8}")
    print("-" * 50)
    for r in results:
        print(f"{r['seed']:>6} | {r['normalized']:>12.8f} | {r['combined_score']:>8.4f} | {r['time']:>7.1f}s")
    
    norms = [r['normalized'] for r in results]
    mean_n = np.mean(norms)
    std_n = np.std(norms)
    print("-" * 50)
    print(f"{'Mean':>6} | {mean_n:>12.8f} +/- {std_n:.8f}")
    print(f"{'Score':>6} | {mean_n / BENCHMARK:>12.6f}")
    
    best = max(results, key=lambda r: r['normalized'])
    print(f"\nBest seed: {best['seed']}, normalized: {best['normalized']:.8f}")
    
    print("\n# For hardcoding in solution.py:")
    print("BEST_POINTS = np.array([")
    for p in best['points']:
        print(f"    [{p[0]:.15f}, {p[1]:.15f}],")
    print("])")
