"""
Heilbronn Triangle (n=11) - Gradient-based optimization via scipy.

The SA approaches keep returning to the known best at 0.03630.
SOTA is 0.03653 (AlphaEvolve). The gap is tiny (~0.6%).

Strategy:
1. Use scipy L-BFGS-B with smooth soft-min objective
2. Parametrize points via barycentric coords (auto-feasible)
3. Use very high beta in soft-min for accuracy
4. Multi-start from known best + perturbations
5. Anneal beta: start low for smooth landscape, increase for precision
"""

import numpy as np
import itertools
import time
from scipy.optimize import minimize
from multiprocessing import Pool

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * abs(V0[0]*(V1[1]-V2[1]) + V1[0]*(V2[1]-V0[1]) + V2[0]*(V0[1]-V1[1]))

TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

KNOWN_BEST = np.array([
    [0.8562136592465739, 0.0000000000000000],
    [0.6448041252020453, 0.6152173017889312],
    [0.2898154821829257, 0.0000000000000000],
    [0.4290718120578101, 0.3298995831194181],
    [0.5759279794357549, 0.1351794251073395],
    [0.5074835537872390, 0.7414106732117386],
    [0.9264251717374268, 0.1274353407089312],
    [0.3640231735946825, 0.6305066317984555],
    [0.1115900853011362, 0.0557385499874744],
    [0.6739217979344588, 0.2973925987695792],
    [0.1354522314961962, 0.2346101469499931],
])


def xy_to_barycentric(points):
    """Convert xy coords to barycentric parameters (u, v) per point.
    Point = u*V0 + v*V1 + (1-u-v)*V2 where u,v >= 0 and u+v <= 1.
    We use softmax parameterization: u = exp(a)/(exp(a)+exp(b)+exp(c)), etc.
    This gives unconstrained optimization over all reals.
    """
    # Solve: x = u*0 + v*1 + w*0.5, y = u*0 + v*0 + w*sqrt3/2
    # where w = 1 - u - v
    # y = w * sqrt3/2 => w = 2*y/sqrt3
    # x = v + 0.5*w => v = x - 0.5*w = x - y/sqrt3
    # u = 1 - v - w
    sqrt3 = np.sqrt(3)
    params = np.zeros(11 * 2)
    for i in range(11):
        x, y = points[i]
        w = 2 * y / sqrt3
        v = x - y / sqrt3
        u = 1 - v - w
        # Clamp to valid
        u = max(1e-10, min(1 - 2e-10, u))
        v = max(1e-10, min(1 - u - 1e-10, v))
        # Store as log-ratio params for unconstrained optimization
        # u = sigmoid(a) * (1 - eps), v = sigmoid(b) * (1 - u - eps)
        # Simpler: just store (u, v) and use bounds
        params[2*i] = u
        params[2*i+1] = v
    return params


def params_to_points(params):
    """Convert (u,v) barycentric params to xy points."""
    points = np.zeros((11, 2))
    for i in range(11):
        u = params[2*i]
        v = params[2*i+1]
        w = 1.0 - u - v
        points[i] = u * V0 + v * V1 + w * V2
    return points


def neg_soft_min(params, beta=300.0):
    """Negative soft-min of triangle areas (minimize this)."""
    points = params_to_points(params)
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    min_a = np.min(areas)
    soft = min_a - (1.0/beta) * np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return -soft


def min_area_from_params(params):
    points = params_to_points(params)
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return np.min(areas)


def run_optimization_task(args):
    """Single optimization task: multi-start L-BFGS-B."""
    task_seed, = args
    rng = np.random.RandomState(task_seed)
    
    best_min = -1
    best_params = None
    
    # Get bounds: u >= eps, v >= eps, u + v <= 1 - eps
    # We handle u+v<=1 via the simplex constraint implicitly
    eps = 1e-8
    bounds = []
    for i in range(11):
        bounds.append((eps, 1.0 - eps))  # u_i
        bounds.append((eps, 1.0 - eps))  # v_i
    
    n_starts = 40
    
    for start in range(n_starts):
        if start == 0:
            # Start from known best
            p0 = xy_to_barycentric(KNOWN_BEST)
        else:
            # Perturbed known best
            p0 = xy_to_barycentric(KNOWN_BEST)
            scale = rng.choice([0.001, 0.005, 0.01, 0.02, 0.05, 0.1])
            p0 += rng.randn(22) * scale
            # Clamp
            for i in range(11):
                p0[2*i] = np.clip(p0[2*i], eps, 1.0 - 2*eps)
                p0[2*i+1] = np.clip(p0[2*i+1], eps, 1.0 - p0[2*i] - eps)
        
        # Anneal beta: optimize with increasing sharpness
        for beta in [50, 150, 500, 2000]:
            try:
                result = minimize(
                    neg_soft_min, p0,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                p0 = result.x.copy()
                # Enforce constraint u + v <= 1
                for i in range(11):
                    u = p0[2*i]
                    v = p0[2*i+1]
                    if u + v > 1 - eps:
                        s = u + v
                        p0[2*i] = u / s * (1 - 2*eps)
                        p0[2*i+1] = v / s * (1 - 2*eps)
            except Exception:
                break
        
        m = min_area_from_params(p0)
        if m > best_min:
            best_min = m
            best_params = p0.copy()
    
    return task_seed, best_params, best_min


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print(f"Gradient-based optimization with {len(seeds)} seeds")
    print(f"Known best normalized: {np.min(np.array([0.5*abs(KNOWN_BEST[i][0]*(KNOWN_BEST[j][1]-KNOWN_BEST[k][1])+KNOWN_BEST[j][0]*(KNOWN_BEST[k][1]-KNOWN_BEST[i][1])+KNOWN_BEST[k][0]*(KNOWN_BEST[i][1]-KNOWN_BEST[j][1])) for i,j,k in itertools.combinations(range(11),3)]))/TRI_AREA:.6f}")
    print()

    t0 = time.time()

    # Generate tasks
    all_tasks = []
    for seed in seeds:
        for i in range(6):
            all_tasks.append((seed * 100 + i,))

    n_workers = min(len(all_tasks), 8)
    print(f"Running {len(all_tasks)} tasks across {n_workers} workers...")

    with Pool(n_workers) as pool:
        all_results = pool.map(run_optimization_task, all_tasks)

    # Group by seed
    results = []
    best_overall_m = -1
    best_overall_pts = None

    for seed in seeds:
        task_seeds = [seed * 100 + i for i in range(6)]
        best_m = -1
        best_p = None
        for ts, params, m in all_results:
            if ts in task_seeds and m > best_m:
                best_m = m
                best_p = params
        
        pts = params_to_points(best_p)
        norm = best_m / TRI_AREA
        results.append({'seed': seed, 'points': pts, 'normalized': norm})
        
        if best_m > best_overall_m:
            best_overall_m = best_m
            best_overall_pts = pts.copy()

    total_time = time.time() - t0

    metrics = [r['normalized'] for r in results]
    mean_m = np.mean(metrics)
    std_m = np.std(metrics)

    print(f"\nTotal time: {total_time:.1f}s\n")
    print(f"| Seed | Metric |")
    print(f"|------|--------|")
    for r in results:
        print(f"| {r['seed']} | {r['normalized']:.6f} |")
    print(f"| **Mean** | **{mean_m:.6f} +/- {std_m:.6f}** |")

    print(f"\nBest normalized: {best_overall_m/TRI_AREA:.6f}")
    print("BEST_POINTS = np.array([")
    for p in best_overall_pts:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
