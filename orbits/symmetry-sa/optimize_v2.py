"""
Heilbronn Triangle (n=11) - Targeted SA with smart perturbations.

Key insight: only perturb points that participate in the smallest triangles.
This focuses the search on the bottleneck rather than wasting moves on
well-separated points.
"""

import numpy as np
import itertools
import time
from multiprocessing import Pool

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * abs(V0[0]*(V1[1]-V2[1]) + V1[0]*(V2[1]-V0[1]) + V2[0]*(V0[1]-V1[1]))
CENTROID = (V0 + V1 + V2) / 3

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


def all_areas(points):
    """Vectorized computation of all 165 triangle areas."""
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )


def min_area(points):
    return np.min(all_areas(points))


def bottleneck_points(points, k=5):
    """Return indices of points involved in the k smallest triangles."""
    areas = all_areas(points)
    smallest_idx = np.argsort(areas)[:k]
    point_set = set()
    for ti in smallest_idx:
        for j in range(3):
            point_set.add(TRIPLET_IDX[ti, j])
    return list(point_set)


def project_to_triangle(x, y):
    sqrt3 = np.sqrt(3)
    y = max(y, 0.0)
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        t = max(0.0, min(0.5, t))
        x, y = t, sqrt3 * t
    if y > sqrt3 * (1 - x):
        t = (x - y / sqrt3 + 1) / 2.0
        t = max(0.5, min(1.0, t))
        x, y = t, sqrt3 * (1 - t)
    y = max(y, 0.0)
    return x, y


def random_point_in_triangle(rng):
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return V0 + r1 * (V1 - V0) + r2 * (V2 - V0)


def targeted_sa(initial_points, seed=42, max_iter=1_000_000,
                T_start=0.005, T_end=1e-8,
                step_start=0.015, step_end=0.0002,
                bottleneck_refresh=5000, bottleneck_bias=0.7):
    """SA that preferentially perturbs bottleneck points."""
    rng = np.random.RandomState(seed)
    points = initial_points.copy()
    n = len(points)

    current_min = min_area(points)
    best_min = current_min
    best_points = points.copy()

    log_T_start = np.log(T_start)
    log_T_end = np.log(T_end)
    log_step_start = np.log(step_start)
    log_step_end = np.log(step_end)

    bn_points = bottleneck_points(points, k=5)

    for it in range(max_iter):
        frac = it / max_iter
        T = np.exp(log_T_start + frac * (log_T_end - log_T_start))
        step = np.exp(log_step_start + frac * (log_step_end - log_step_start))

        # Refresh bottleneck points periodically
        if it % bottleneck_refresh == 0 and it > 0:
            bn_points = bottleneck_points(points, k=5)

        # With high probability, pick from bottleneck points
        if rng.random() < bottleneck_bias and len(bn_points) > 0:
            idx = bn_points[rng.randint(len(bn_points))]
        else:
            idx = rng.randint(n)

        old_x, old_y = points[idx].copy()

        dx = rng.randn() * step
        dy = rng.randn() * step
        new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)

        points[idx] = [new_x, new_y]
        new_min = min_area(points)

        delta = new_min - current_min

        if delta > 0 or rng.random() < np.exp(delta / T):
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_points = points.copy()
        else:
            points[idx] = [old_x, old_y]

    return best_points, best_min


def run_task(args):
    """Single optimization task."""
    task_seed, perturb_scale = args

    rng = np.random.RandomState(task_seed)
    
    # Perturb known best
    init = KNOWN_BEST.copy()
    for i in range(11):
        dx = rng.randn() * perturb_scale
        dy = rng.randn() * perturb_scale
        init[i, 0], init[i, 1] = project_to_triangle(
            init[i, 0] + dx, init[i, 1] + dy)

    # Phase 1: exploration with standard SA
    pts, m = targeted_sa(
        init, seed=task_seed,
        max_iter=500_000,
        T_start=0.008, T_end=1e-7,
        step_start=0.02, step_end=0.0005,
        bottleneck_bias=0.6
    )

    # Phase 2: intensive refinement
    pts2, m2 = targeted_sa(
        pts, seed=task_seed + 500000,
        max_iter=500_000,
        T_start=0.002, T_end=1e-9,
        step_start=0.005, step_end=0.00005,
        bottleneck_bias=0.8
    )

    best_pts = pts2 if m2 > m else pts
    best_m = max(m, m2)

    return task_seed, best_pts, best_m


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print(f"Targeted SA optimization with {len(seeds)} seeds")
    print(f"Known best normalized: {min_area(KNOWN_BEST)/TRI_AREA:.6f}")
    print()

    t0 = time.time()

    # Generate tasks: various perturbation scales per seed
    all_tasks = []
    for seed in seeds:
        for i, scale in enumerate([0.0, 0.002, 0.005, 0.01, 0.02, 0.03]):
            all_tasks.append((seed * 100 + i, scale))

    n_workers = min(len(all_tasks), 8)
    print(f"Running {len(all_tasks)} tasks across {n_workers} workers...")

    with Pool(n_workers) as pool:
        all_results = pool.map(run_task, all_tasks)

    # Group by seed
    seed_map = {}
    for seed in seeds:
        task_seeds = [seed * 100 + i for i in range(6)]
        best_m = -1
        best_pts = None
        for ts, pts, m in all_results:
            if ts in task_seeds and m > best_m:
                best_m = m
                best_pts = pts.copy()
        seed_map[seed] = (best_pts, best_m)

    total_time = time.time() - t0

    print(f"\nTotal time: {total_time:.1f}s\n")
    print(f"| Seed | Metric |")
    print(f"|------|--------|")

    metrics = []
    best_overall_pts = None
    best_overall_m = -1

    for seed in seeds:
        pts, m = seed_map[seed]
        norm = m / TRI_AREA
        metrics.append(norm)
        print(f"| {seed} | {norm:.6f} |")
        if m > best_overall_m:
            best_overall_m = m
            best_overall_pts = pts.copy()

    mean_m = np.mean(metrics)
    std_m = np.std(metrics)
    print(f"| **Mean** | **{mean_m:.6f} +/- {std_m:.6f}** |")

    print(f"\nBest normalized: {best_overall_m/TRI_AREA:.6f}")
    print("BEST_POINTS = np.array([")
    for p in best_overall_pts:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
