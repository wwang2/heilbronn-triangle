"""
V4: Two strategies combined:
1. Differential evolution on smooth objective (global search)
2. Intensive local perturbation + Nelder-Mead around best known solutions
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

# Best known from V2
KNOWN_BEST = np.array([
    [0.573056295873995, 0.160321182152447],
    [0.967036553375919, 0.057094364345492],
    [0.836695199483545, 0.000000000000000],
    [0.634410370885316, 0.633219812346895],
    [0.434481748964236, 0.358671650948760],
    [0.117511775643195, 0.203536365901648],
    [0.098788311356975, 0.055533383138888],
    [0.378638275082370, 0.655820730132905],
    [0.507650639215252, 0.762585451097057],
    [0.294450643417061, 0.000000000000000],
    [0.659845029066328, 0.318694842803040],
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


def softmin_lse(areas, beta):
    scaled = -beta * areas
    mx = np.max(scaled)
    return -(1.0 / beta) * (mx + np.log(np.sum(np.exp(scaled - mx))))


def objective_projected(x, beta):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -softmin_lse(areas, beta)


def objective_true(x):
    points = project_vec(x.reshape(11, 2))
    areas = triangle_areas(points)
    return -np.min(areas)


def random_tri_points(n, rng):
    points = np.empty((n, 2))
    for i in range(n):
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        points[i] = V0 + r1 * (V1 - V0) + r2 * (V2 - V0)
    return points


class TakeStep:
    def __init__(self, stepsize, rng):
        self.stepsize = stepsize
        self.rng = rng
    def __call__(self, x):
        x += self.rng.uniform(-self.stepsize, self.stepsize, size=x.shape)
        points = project_vec(x.reshape(11, 2))
        x[:] = points.flatten()
        return x


def polish_nelder_mead(points, n_iter=5000):
    from scipy.optimize import minimize
    x0 = points.flatten()
    result = minimize(
        objective_true, x0,
        method='Nelder-Mead',
        options={'maxiter': n_iter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True},
    )
    return project_vec(result.x.reshape(11, 2))


def run_de_seed(seed):
    """Differential evolution approach."""
    from scipy.optimize import differential_evolution, basinhopping
    
    t0 = time.time()
    rng = np.random.RandomState(seed)
    
    best_min_area = -np.inf
    best_points = None
    
    # Strategy 1: DE on smooth objective with moderate beta
    for beta in [50, 200, 500]:
        bounds = [(0, 1)] * 22  # 11 points * 2 coords
        
        result = differential_evolution(
            objective_projected,
            bounds=bounds,
            args=(beta,),
            seed=int(rng.randint(0, 2**31)),
            maxiter=300,
            popsize=30,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.9,
            strategy='best1bin',
            disp=False,
        )
        
        pts = project_vec(result.x.reshape(11, 2))
        pts = polish_nelder_mead(pts, 5000)
        areas = triangle_areas(pts)
        ma = np.min(areas)
        if ma > best_min_area:
            best_min_area = ma
            best_points = pts.copy()
    
    # Strategy 2: Basin-hopping from best DE result
    current_x = best_points.flatten()
    schedule = [(200, 30, 0.05), (800, 20, 0.02), (3000, 15, 0.01)]
    for beta, n_iter, stepsize in schedule:
        take_step = TakeStep(stepsize, rng)
        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "args": (beta,),
            "options": {"maxiter": 500, "ftol": 1e-15, "gtol": 1e-12},
        }
        result = basinhopping(
            objective_projected, current_x,
            minimizer_kwargs=minimizer_kwargs,
            niter=n_iter,
            take_step=take_step,
            seed=int(rng.randint(0, 2**31)),
            disp=False,
        )
        current_x = result.x.copy()
    
    pts = project_vec(current_x.reshape(11, 2))
    pts = polish_nelder_mead(pts, 5000)
    areas = triangle_areas(pts)
    ma = np.min(areas)
    if ma > best_min_area:
        best_min_area = ma
        best_points = pts.copy()
    
    # Strategy 3: Intensive local search around KNOWN_BEST
    for _ in range(100):
        scale = rng.choice([0.003, 0.005, 0.01, 0.02, 0.04])
        perturbed = KNOWN_BEST + rng.uniform(-scale, scale, size=KNOWN_BEST.shape)
        perturbed = project_vec(perturbed)
        polished = polish_nelder_mead(perturbed, 5000)
        areas = triangle_areas(polished)
        ma = np.min(areas)
        if ma > best_min_area:
            best_min_area = ma
            best_points = polished.copy()
    
    # Strategy 4: Local search around current best
    for _ in range(100):
        scale = rng.choice([0.002, 0.005, 0.01, 0.03])
        perturbed = best_points + rng.uniform(-scale, scale, size=best_points.shape)
        perturbed = project_vec(perturbed)
        polished = polish_nelder_mead(perturbed, 5000)
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
    
    print(f"Running V4 (DE + local search) with {len(seeds)} seeds in parallel...")
    print(f"SOTA benchmark: {BENCHMARK}")
    print()
    
    with Pool(len(seeds)) as p:
        results = p.map(run_de_seed, seeds)
    
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
