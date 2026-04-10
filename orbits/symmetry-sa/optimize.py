"""
Symmetry-aware Simulated Annealing for Heilbronn Triangle (n=11).

Strategy:
1. Start from known best config + symmetry-generated seeds + random seeds
2. Vectorized area computation  
3. Flat parallelism: all (seed, config) pairs in one Pool
4. Hybrid: symmetric seeds -> unconstrained SA refinement
"""

import numpy as np
import itertools
import time
import sys
from multiprocessing import Pool

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
CENTROID = (V0 + V1 + V2) / 3
TRI_AREA = 0.5 * abs(V0[0]*(V1[1]-V2[1]) + V1[0]*(V2[1]-V0[1]) + V2[0]*(V0[1]-V1[1]))

# Precompute triplet indices for n=11
TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

# Known best from literature (normalized ~0.03630)
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


def min_area(points):
    """Vectorized min triangle area."""
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return np.min(areas)


def project_to_triangle(x, y):
    """Project point to inside equilateral triangle."""
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
    """Sample uniform random point in equilateral triangle."""
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return V0 + r1 * (V1 - V0) + r2 * (V2 - V0)


def rotate_about_centroid(point, angle):
    """Rotate point by angle (radians) about centroid."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = point[0] - CENTROID[0]
    dy = point[1] - CENTROID[1]
    return np.array([
        CENTROID[0] + dx * cos_a - dy * sin_a,
        CENTROID[1] + dx * sin_a + dy * cos_a
    ])


def generate_c3_config(rng):
    """Generate a C3-symmetric config: 3 groups of 3 + 2 special points."""
    points = np.zeros((11, 2))
    points[0] = CENTROID
    t = rng.uniform(0.1, 0.8)
    points[1] = np.array([0.5, t * np.sqrt(3) / 2])

    for g in range(3):
        base = random_point_in_triangle(rng)
        for _ in range(10):
            if np.linalg.norm(base - CENTROID) > 0.1:
                break
            base = random_point_in_triangle(rng)
        p1 = rotate_about_centroid(base, 2*np.pi/3)
        p2 = rotate_about_centroid(base, 4*np.pi/3)
        p1[0], p1[1] = project_to_triangle(p1[0], p1[1])
        p2[0], p2[1] = project_to_triangle(p2[0], p2[1])
        points[2 + g * 3] = base
        points[3 + g * 3] = p1
        points[4 + g * 3] = p2

    return points


def simulated_annealing(initial_points, seed=42, max_iter=500_000,
                         T_start=0.02, T_end=1e-7, step_start=0.05, step_end=0.001):
    """Run SA to maximize min triangle area."""
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

    for it in range(max_iter):
        frac = it / max_iter
        T = np.exp(log_T_start + frac * (log_T_end - log_T_start))
        step = np.exp(log_step_start + frac * (log_step_end - log_step_start))

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


def run_single_config(args):
    """Run SA from a single starting config with restarts.
    Returns (master_seed, best_points, best_min_area)."""
    master_seed, inner_seed, config_type = args

    best_overall_points = None
    best_overall_min = -1

    n_restarts = 8

    for restart in range(n_restarts):
        restart_seed = inner_seed * 1000 + restart
        rng_restart = np.random.RandomState(restart_seed)

        if config_type == 'known':
            init = KNOWN_BEST.copy()
            perturb_scale = rng_restart.choice([0.005, 0.01, 0.02, 0.04])
            for i in range(11):
                dx = rng_restart.randn() * perturb_scale
                dy = rng_restart.randn() * perturb_scale
                init[i, 0], init[i, 1] = project_to_triangle(
                    init[i, 0] + dx, init[i, 1] + dy)
        elif config_type == 'c3':
            init = generate_c3_config(rng_restart)
        else:
            init = np.array([random_point_in_triangle(rng_restart) for _ in range(11)])

        pts, m = simulated_annealing(
            init, seed=restart_seed,
            max_iter=300_000,
            T_start=0.01, T_end=1e-7,
            step_start=0.03, step_end=0.0005
        )

        if m > best_overall_min:
            best_overall_min = m
            best_overall_points = pts.copy()

    # Final long refinement from best found
    pts, m = simulated_annealing(
        best_overall_points, seed=inner_seed + 999999,
        max_iter=500_000,
        T_start=0.003, T_end=1e-8,
        step_start=0.008, step_end=0.0001
    )
    if m > best_overall_min:
        best_overall_min = m
        best_overall_points = pts.copy()

    return master_seed, best_overall_points, best_overall_min


if __name__ == '__main__':
    seeds = [42, 123, 7]

    print(f"Running optimization with {len(seeds)} seeds...")
    print(f"Known best normalized: 0.03630")
    print()

    t0_total = time.time()

    # Build flat list of all (master_seed, inner_seed, config_type) tasks
    all_tasks = []
    for seed in seeds:
        all_tasks.extend([
            (seed, seed, 'known'),
            (seed, seed + 100, 'known'),
            (seed, seed + 200, 'c3'),
            (seed, seed + 300, 'random'),
            (seed, seed + 400, 'known'),
            (seed, seed + 500, 'c3'),
            (seed, seed + 600, 'known'),
            (seed, seed + 700, 'random'),
        ])

    # Run all tasks in one flat pool
    n_workers = min(len(all_tasks), 8)
    print(f"Running {len(all_tasks)} tasks across {n_workers} workers...")
    with Pool(n_workers) as pool:
        all_results = pool.map(run_single_config, all_tasks)

    # Group by master_seed
    seed_results = {s: [] for s in seeds}
    for master_seed, pts, m in all_results:
        seed_results[master_seed].append((pts, m))

    results = []
    for seed in seeds:
        best_pts = None
        best_m = -1
        for pts, m in seed_results[seed]:
            if m > best_m:
                best_m = m
                best_pts = pts.copy()
        normalized = best_m / TRI_AREA
        results.append({
            'points': best_pts,
            'min_area': best_m,
            'normalized': normalized,
            'seed': seed,
        })
        print(f"  Seed {seed}: normalized={normalized:.6f}")

    total_time = time.time() - t0_total

    metrics = [r['normalized'] for r in results]
    mean_m = np.mean(metrics)
    std_m = np.std(metrics)

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\nResults:")
    print(f"| Seed | Metric | ")
    print(f"|------|--------|")
    for r in results:
        print(f"| {r['seed']} | {r['normalized']:.6f} |")
    print(f"| **Mean** | **{mean_m:.6f} +/- {std_m:.6f}** |")

    # Save best
    best_result = max(results, key=lambda r: r['normalized'])
    print(f"\nBest config (seed={best_result['seed']}, metric={best_result['normalized']:.6f}):")
    print("BEST_POINTS = np.array([")
    for p in best_result['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
