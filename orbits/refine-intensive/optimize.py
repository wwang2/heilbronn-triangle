"""
Intensive Simulated Annealing for Heilbronn Triangle n=11 with Numba JIT.

Strategy:
- Numba-compiled inner loop for ~100x speedup over pure Python
- Incremental area updates: when moving point i, only recompute C(10,2)=45 triplets
- Multi-phase cooling with reheating
- Very fine perturbations (0.02 -> 0.0002) for the final polish
- 200M iterations per seed across 6 phases
- Multiple restarts from the parent's best configuration
"""
import numpy as np
from numba import njit
import multiprocessing as mp
import time
import json
import sys
import os

# Equilateral triangle
SQRT3_HALF = np.sqrt(3) / 2.0
TRI_AREA = 0.5 * SQRT3_HALF  # area of equilateral triangle

# Precompute all C(11,3)=165 triplet indices
N = 11
_triplets = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            _triplets.append((i, j, k))
TRIPLETS = np.array(_triplets, dtype=np.int32)

# For each point i, precompute which triplet indices involve it
POINT_TRIPLETS = []
for pt in range(N):
    indices = []
    for t_idx in range(len(_triplets)):
        if pt in _triplets[t_idx]:
            indices.append(t_idx)
    POINT_TRIPLETS.append(np.array(indices, dtype=np.int32))

# Parent's best configuration
PARENT_POINTS = np.array([
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


@njit(cache=True)
def triangle_area(px, py, qx, qy, rx, ry):
    """Compute area of triangle given 3 points."""
    return 0.5 * abs(px * (qy - ry) + qx * (ry - py) + rx * (py - qy))


@njit(cache=True)
def compute_all_areas(points, triplets):
    """Compute all 165 triangle areas."""
    n_tri = triplets.shape[0]
    areas = np.empty(n_tri)
    for t in range(n_tri):
        i, j, k = triplets[t, 0], triplets[t, 1], triplets[t, 2]
        areas[t] = triangle_area(
            points[i, 0], points[i, 1],
            points[j, 0], points[j, 1],
            points[k, 0], points[k, 1]
        )
    return areas


@njit(cache=True)
def is_inside_triangle(x, y):
    """Check if point (x,y) is inside equilateral triangle (0,0)-(1,0)-(0.5, sqrt(3)/2)."""
    sqrt3 = 1.7320508075688772
    tol = 1e-9
    if y < -tol:
        return False
    if sqrt3 * x > sqrt3 - y + tol:
        return False
    if y > sqrt3 * x + tol:
        return False
    return True


@njit(cache=True)
def sa_inner(points, triplets, point_triplet_indices, point_triplet_counts,
             max_point_triplets, n_iterations, temp_start, temp_end,
             perturb_start, perturb_end, seed, focus_prob=0.4):
    """
    Core SA loop, fully JIT compiled.
    """
    np.random.seed(seed)
    n_points = points.shape[0]
    n_tri = triplets.shape[0]

    # Compute initial areas
    areas = compute_all_areas(points, triplets)
    min_area = areas.min()

    best_points = points.copy()
    best_min_area = min_area

    # Find the minimum area triplet index
    min_idx = 0
    for t in range(n_tri):
        if areas[t] < areas[min_idx]:
            min_idx = t

    accepted = 0
    improved = 0

    log_ratio = np.log(temp_end / temp_start)
    log_ratio_p = np.log(perturb_end / perturb_start)
    inv_n = 1.0 / max(n_iterations - 1, 1)

    for iteration in range(n_iterations):
        # Adaptive temperature (exponential cooling)
        frac = iteration * inv_n
        temp = temp_start * np.exp(log_ratio * frac)
        perturb = perturb_start * np.exp(log_ratio_p * frac)

        # Choose which point to move
        if np.random.random() < focus_prob:
            choice = int(np.random.random() * 3)
            pt_idx = triplets[min_idx, choice]
        else:
            pt_idx = int(np.random.random() * n_points)

        # Save old position
        old_x = points[pt_idx, 0]
        old_y = points[pt_idx, 1]

        # Perturb
        new_x = old_x + np.random.randn() * perturb
        new_y = old_y + np.random.randn() * perturb

        # Check bounds
        if not is_inside_triangle(new_x, new_y):
            continue

        # Apply move temporarily
        points[pt_idx, 0] = new_x
        points[pt_idx, 1] = new_y

        # Recompute only affected triplets
        old_areas_backup = np.empty(max_point_triplets)
        n_affected = point_triplet_counts[pt_idx]

        base = pt_idx * max_point_triplets
        for q in range(n_affected):
            t = point_triplet_indices[base + q]
            old_areas_backup[q] = areas[t]
            i, j, k = triplets[t, 0], triplets[t, 1], triplets[t, 2]
            areas[t] = triangle_area(
                points[i, 0], points[i, 1],
                points[j, 0], points[j, 1],
                points[k, 0], points[k, 1]
            )

        # Find new global min
        new_min_area = areas[0]
        new_min_idx = 0
        for t in range(1, n_tri):
            if areas[t] < new_min_area:
                new_min_area = areas[t]
                new_min_idx = t

        # Acceptance criterion
        delta = new_min_area - min_area
        accept = False
        if delta >= 0:
            accept = True
        elif temp > 1e-30:
            if np.random.random() < np.exp(delta / temp):
                accept = True

        if accept:
            min_area = new_min_area
            min_idx = new_min_idx
            accepted += 1
            if new_min_area > best_min_area:
                best_min_area = new_min_area
                for i in range(n_points):
                    best_points[i, 0] = points[i, 0]
                    best_points[i, 1] = points[i, 1]
                improved += 1
        else:
            # Revert
            points[pt_idx, 0] = old_x
            points[pt_idx, 1] = old_y
            for q in range(n_affected):
                t = point_triplet_indices[base + q]
                areas[t] = old_areas_backup[q]

    return best_points, best_min_area, accepted, improved


def build_point_triplet_data():
    """Build flat arrays for point-triplet mapping (Numba-friendly)."""
    max_pt = max(len(pt) for pt in POINT_TRIPLETS)
    flat = np.zeros(N * max_pt, dtype=np.int32)
    counts = np.zeros(N, dtype=np.int32)
    for pt in range(N):
        indices = POINT_TRIPLETS[pt]
        counts[pt] = len(indices)
        base = pt * max_pt
        for q in range(len(indices)):
            flat[base + q] = indices[q]
    return flat, counts, max_pt


def run_optimization(seed, n_phases=6, iters_per_phase=50_000_000):
    """Run multi-phase SA optimization from parent config."""
    t0 = time.time()
    rng = np.random.RandomState(seed)

    flat_pt_tri, pt_counts, max_pt = build_point_triplet_data()

    # Start from parent best
    points = PARENT_POINTS.copy()
    best_points = points.copy()
    best_metric = compute_all_areas(points, TRIPLETS).min() / TRI_AREA

    phase_configs = [
        # (temp_start, temp_end, perturb_start, perturb_end, iters, focus_prob)
        (2e-3,  1e-5,  0.03,   0.005,  iters_per_phase, 0.30),    # Phase 1: broad exploration
        (1e-3,  1e-6,  0.015,  0.003,  iters_per_phase, 0.35),    # Phase 2: medium
        (5e-4,  1e-7,  0.008,  0.001,  iters_per_phase, 0.40),    # Phase 3: fine
        (2e-4,  1e-8,  0.004,  0.0005, iters_per_phase, 0.45),    # Phase 4: very fine
        (1e-4,  1e-9,  0.002,  0.0003, iters_per_phase, 0.50),    # Phase 5: ultra-fine
        (5e-5,  1e-10, 0.001,  0.0001, iters_per_phase, 0.55),    # Phase 6: polish
    ]

    for phase_idx in range(n_phases):
        t_start, t_end, p_start, p_end, n_iters, focus = phase_configs[phase_idx]
        phase_seed = rng.randint(0, 2**31)

        # Start each phase from the best found so far
        points = best_points.copy()

        result_pts, result_min, acc, imp = sa_inner(
            points, TRIPLETS, flat_pt_tri, pt_counts, max_pt,
            n_iters, t_start, t_end, p_start, p_end, phase_seed, focus
        )

        metric = result_min / TRI_AREA
        if metric > best_metric:
            best_metric = metric
            best_points = result_pts.copy()

        elapsed = time.time() - t0
        print(f"  Seed {seed} Phase {phase_idx+1}/{n_phases}: metric={metric:.10f} "
              f"(best={best_metric:.10f}) acc={acc} imp={imp} time={elapsed:.1f}s",
              flush=True)

    elapsed = time.time() - t0
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed}


def run_with_seed(seed):
    """Wrapper for multiprocessing."""
    result = run_optimization(seed, n_phases=6, iters_per_phase=50_000_000)
    result['points'] = result['points'].tolist()
    return result


def main():
    seeds = [42, 123, 7]

    print("Warming up Numba JIT...")
    flat_pt_tri, pt_counts, max_pt = build_point_triplet_data()
    pts_warmup = PARENT_POINTS.copy()
    sa_inner(pts_warmup, TRIPLETS, flat_pt_tri, pt_counts, max_pt,
             100, 1e-3, 1e-6, 0.01, 0.001, 0, 0.4)
    print("JIT compilation done.\n")

    print(f"Running {len(seeds)} seeds in parallel (6 phases x 50M iters each = 300M per seed)...")
    t0 = time.time()

    with mp.Pool(len(seeds)) as pool:
        results = pool.map(run_with_seed, seeds)

    total_time = time.time() - t0

    # Sort by metric
    results.sort(key=lambda r: r['metric'], reverse=True)

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\nResults:")
    print(f"| Seed | Metric | Time |")
    print(f"|------|--------|------|")
    metrics = []
    for r in results:
        print(f"| {r['seed']} | {r['metric']:.10f} | {r['time']:.1f}s |")
        metrics.append(r['metric'])

    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    print(f"| **Mean** | **{mean_metric:.6f} +/- {std_metric:.6f}** | |")

    best = results[0]
    print(f"\nBest: seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"\nBest points:")
    for i, (x, y) in enumerate(best['points']):
        print(f"    [{x:.16f}, {y:.16f}],")

    # Save results
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
    with open(save_path, 'w') as f:
        json.dump({
            'best_metric': best['metric'],
            'best_points': best['points'],
            'best_seed': best['seed'],
            'all_results': [{'seed': r['seed'], 'metric': r['metric'], 'time': r['time']} for r in results],
            'mean_metric': mean_metric,
            'std_metric': std_metric,
        }, f, indent=2)

    return results


if __name__ == '__main__':
    main()
