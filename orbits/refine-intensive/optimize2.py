"""
Round 2: Ultra-fine refinement starting from the round 1 best (0.036529).
Even more iterations, even finer perturbations.
"""
import numpy as np
from numba import njit
import multiprocessing as mp
import time
import json
import sys
import os

SQRT3_HALF = np.sqrt(3) / 2.0
TRI_AREA = 0.5 * SQRT3_HALF

N = 11
_triplets = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            _triplets.append((i, j, k))
TRIPLETS = np.array(_triplets, dtype=np.int32)

POINT_TRIPLETS = []
for pt in range(N):
    indices = []
    for t_idx in range(len(_triplets)):
        if pt in _triplets[t_idx]:
            indices.append(t_idx)
    POINT_TRIPLETS.append(np.array(indices, dtype=np.int32))

# Round 1 best (seed=123, metric=0.0365290663)
BEST_POINTS = np.array([
    [0.1062251314076116, 0.0710718934658221],
    [0.8521731052225914, 0.2560435711332299],
    [0.4999554613307710, 0.2111139263237521],
    [0.2774109177361259, 0.0000014741238322],
    [0.1478077583239472, 0.2560103184540150],
    [0.4279849010343168, 0.7412904547699716],
    [0.5720290867420540, 0.7412668049527609],
    [0.8937693919848929, 0.0710706330075633],
    [0.4093040548459591, 0.4392242143531872],
    [0.7225360221512087, 0.0000007258203650],
    [0.5906546591209809, 0.4393056854396173],
])


@njit(cache=True)
def triangle_area(px, py, qx, qy, rx, ry):
    return 0.5 * abs(px * (qy - ry) + qx * (ry - py) + rx * (py - qy))


@njit(cache=True)
def compute_all_areas(points, triplets):
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
    np.random.seed(seed)
    n_points = points.shape[0]
    n_tri = triplets.shape[0]

    areas = compute_all_areas(points, triplets)
    min_area = areas.min()

    best_points = points.copy()
    best_min_area = min_area

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
        frac = iteration * inv_n
        temp = temp_start * np.exp(log_ratio * frac)
        perturb = perturb_start * np.exp(log_ratio_p * frac)

        if np.random.random() < focus_prob:
            choice = int(np.random.random() * 3)
            pt_idx = triplets[min_idx, choice]
        else:
            pt_idx = int(np.random.random() * n_points)

        old_x = points[pt_idx, 0]
        old_y = points[pt_idx, 1]

        new_x = old_x + np.random.randn() * perturb
        new_y = old_y + np.random.randn() * perturb

        if not is_inside_triangle(new_x, new_y):
            continue

        points[pt_idx, 0] = new_x
        points[pt_idx, 1] = new_y

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

        new_min_area = areas[0]
        new_min_idx = 0
        for t in range(1, n_tri):
            if areas[t] < new_min_area:
                new_min_area = areas[t]
                new_min_idx = t

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
            points[pt_idx, 0] = old_x
            points[pt_idx, 1] = old_y
            for q in range(n_affected):
                t = point_triplet_indices[base + q]
                areas[t] = old_areas_backup[q]

    return best_points, best_min_area, accepted, improved


def build_point_triplet_data():
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


def run_optimization(seed, n_phases=8, iters_per_phase=100_000_000):
    """Run ultra-fine SA from round 1 best."""
    t0 = time.time()
    rng = np.random.RandomState(seed)
    flat_pt_tri, pt_counts, max_pt = build_point_triplet_data()

    points = BEST_POINTS.copy()
    best_points = points.copy()
    best_metric = compute_all_areas(points, TRIPLETS).min() / TRI_AREA

    phase_configs = [
        # (temp_start, temp_end, perturb_start, perturb_end, iters, focus_prob)
        (5e-4,  1e-6,  0.01,    0.002,   iters_per_phase, 0.35),
        (2e-4,  1e-7,  0.005,   0.001,   iters_per_phase, 0.40),
        (1e-4,  1e-8,  0.003,   0.0005,  iters_per_phase, 0.45),
        (5e-5,  1e-9,  0.002,   0.0003,  iters_per_phase, 0.50),
        (2e-5,  1e-10, 0.001,   0.0001,  iters_per_phase, 0.50),
        (1e-5,  1e-11, 0.0005,  0.00005, iters_per_phase, 0.55),
        (5e-6,  1e-12, 0.0003,  0.00002, iters_per_phase, 0.55),
        (2e-6,  1e-13, 0.0001,  0.00001, iters_per_phase, 0.60),
    ]

    for phase_idx in range(n_phases):
        t_start, t_end, p_start, p_end, n_iters, focus = phase_configs[phase_idx]
        phase_seed = rng.randint(0, 2**31)

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
    result = run_optimization(seed, n_phases=8, iters_per_phase=100_000_000)
    result['points'] = result['points'].tolist()
    return result


def main():
    seeds = [42, 123, 7]

    print("Warming up Numba JIT...")
    flat_pt_tri, pt_counts, max_pt = build_point_triplet_data()
    pts_warmup = BEST_POINTS.copy()
    sa_inner(pts_warmup, TRIPLETS, flat_pt_tri, pt_counts, max_pt,
             100, 1e-3, 1e-6, 0.01, 0.001, 0, 0.4)
    print("JIT compilation done.\n")

    print(f"Running {len(seeds)} seeds (8 phases x 100M iters = 800M per seed)...")
    t0 = time.time()

    with mp.Pool(len(seeds)) as pool:
        results = pool.map(run_with_seed, seeds)

    total_time = time.time() - t0

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
    print(f"| **Mean** | **{mean_metric:.10f} +/- {std_metric:.10f}** | |")

    best = results[0]
    print(f"\nBest: seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"\nBest points:")
    for i, (x, y) in enumerate(best['points']):
        print(f"    [{x:.16f}, {y:.16f}],")

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results2.json')
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
