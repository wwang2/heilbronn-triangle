"""
V5: Ultra-intensive local search around the best known solution.
Focus all compute on polishing the local optimum.
"""

import numpy as np
from itertools import combinations
from multiprocessing import Pool
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2
EQUIL_AREA = np.sqrt(3) / 4
BENCHMARK = 0.036529889880030156
TRIPLETS = np.array(list(combinations(range(11), 3)))

KNOWN_BEST = np.array([
    [0.571410114990977, 0.160903132947647],
    [0.967840603269951, 0.055701709077209],
    [0.834572069630165, 0.000007622012137],
    [0.634431916255954, 0.633182494670282],
    [0.433427347279021, 0.359597997163961],
    [0.118473366889313, 0.205201890796039],
    [0.098922062316121, 0.055790652902838],
    [0.378869492469816, 0.656221210395555],
    [0.507327904535471, 0.763089164415980],
    [0.293006043913453, 0.000051847694994],
    [0.657443788714565, 0.321098845590150],
])


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


def objective_true(x):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -np.min(areas)


def softmin_lse(areas, beta):
    scaled = -beta * areas
    mx = np.max(scaled)
    return -(1.0 / beta) * (mx + np.log(np.sum(np.exp(scaled - mx))))


def objective_smooth(x, beta):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -softmin_lse(areas, beta)


def polish_nelder_mead(points, n_iter=10000):
    from scipy.optimize import minimize
    x0 = points.flatten()
    result = minimize(
        objective_true, x0,
        method='Nelder-Mead',
        options={'maxiter': n_iter, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True},
    )
    return project_vec(result.x.reshape(11, 2))


def polish_powell(points, n_iter=10000):
    from scipy.optimize import minimize
    x0 = points.flatten()
    result = minimize(
        objective_true, x0,
        method='Powell',
        options={'maxiter': n_iter, 'ftol': 1e-15, 'xtol': 1e-13},
    )
    return project_vec(result.x.reshape(11, 2))


def polish_lbfgsb(points, beta=2000):
    from scipy.optimize import minimize
    x0 = points.flatten()
    result = minimize(
        objective_smooth, x0,
        args=(beta,),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14},
    )
    return project_vec(result.x.reshape(11, 2))


def run_local_search(seed):
    """Intensive local search around known best."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    
    best_min_area = -np.inf
    best_points = None
    
    # Start from known best
    current = KNOWN_BEST.copy()
    areas = triangle_areas(current)
    best_min_area = np.min(areas)
    best_points = current.copy()
    
    # Strategy 1: Multi-scale perturbation + multiple polish methods
    scales = [0.001, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]
    
    for iteration in range(500):
        scale = rng.choice(scales)
        
        # Perturbation strategies
        strategy = rng.randint(4)
        if strategy == 0:
            # Perturb all points
            perturbed = best_points + rng.uniform(-scale, scale, size=best_points.shape)
        elif strategy == 1:
            # Perturb only a few random points
            perturbed = best_points.copy()
            n_perturb = rng.randint(1, 5)
            idx = rng.choice(11, size=n_perturb, replace=False)
            perturbed[idx] += rng.uniform(-scale, scale, size=(n_perturb, 2))
        elif strategy == 2:
            # Gaussian perturbation
            perturbed = best_points + rng.normal(0, scale/2, size=best_points.shape)
        else:
            # Swap-inspired: move one point toward centroid of others
            perturbed = best_points.copy()
            idx = rng.randint(11)
            others = np.delete(best_points, idx, axis=0)
            centroid = others.mean(axis=0)
            alpha = rng.uniform(-0.3, 0.3)
            perturbed[idx] = best_points[idx] + alpha * (centroid - best_points[idx])
            perturbed[idx] += rng.uniform(-scale/2, scale/2, size=2)
        
        perturbed = project_vec(perturbed)
        
        # Choose polish method
        polish = rng.randint(3)
        if polish == 0:
            polished = polish_nelder_mead(perturbed, n_iter=8000)
        elif polish == 1:
            polished = polish_powell(perturbed, n_iter=8000)
        else:
            # L-BFGS-B smooth then Nelder-Mead true
            polished = polish_lbfgsb(perturbed, beta=3000)
            polished = polish_nelder_mead(polished, n_iter=5000)
        
        areas = triangle_areas(polished)
        ma = np.min(areas)
        if ma > best_min_area:
            best_min_area = ma
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
    
    print(f"Running V5 (intensive local search) with {len(seeds)} seeds in parallel...")
    print(f"SOTA benchmark: {BENCHMARK}")
    print()
    
    with Pool(len(seeds)) as p:
        results = p.map(run_local_search, seeds)
    
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
