"""
Ultra-fine refinement for Heilbronn Triangle (n=11).

Current: 0.03652980, SOTA: 0.03652989. Gap: ~0.0000001.

Key optimization: incremental area computation. When moving point i,
only 45 out of 165 triplets change. Cache the other 120 areas.
"""

import numpy as np
import itertools
import time
from numba import njit
from multiprocessing import Pool

TRI_AREA = np.sqrt(3) / 4
TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

REFINE_BEST = np.array([
    [0.1062293556864848, 0.0710758645911361],
    [0.8521738851544579, 0.2560423324049147],
    [0.4999923178324033, 0.2111247288075353],
    [0.2774464405065116, 0.0000000295515091],
    [0.1478227903234471, 0.2560365776868428],
    [0.4279868509264755, 0.7412948545022972],
    [0.5720190466372811, 0.7412846410657106],
    [0.8937681931796337, 0.0710758658108138],
    [0.4093311687853985, 0.4392800602323884],
    [0.7225448801260982, 0.0000000310705539],
    [0.5906639243724431, 0.4392968836889751],
])


@njit(cache=True)
def project_to_triangle(x, y):
    sqrt3 = 1.7320508075688772
    if y < 0.0:
        y = 0.0
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        if t < 0.0: t = 0.0
        elif t > 0.5: t = 0.5
        x = t
        y = sqrt3 * t
    if y > sqrt3 * (1.0 - x):
        t = (x - y / sqrt3 + 1.0) / 2.0
        if t < 0.5: t = 0.5
        elif t > 1.0: t = 1.0
        x = t
        y = sqrt3 * (1.0 - t)
    if y < 0.0:
        y = 0.0
    return x, y


@njit(cache=True)
def build_point_triplet_map(triplet_idx, n_points):
    """For each point, list of triplet indices involving it."""
    # Each point is in C(10,2) = 45 triplets
    pt_map = np.empty((n_points, 45), dtype=np.int32)
    pt_count = np.zeros(n_points, dtype=np.int32)
    for t in range(len(triplet_idx)):
        for j in range(3):
            p = triplet_idx[t, j]
            pt_map[p, pt_count[p]] = t
            pt_count[p] += 1
    return pt_map, pt_count


@njit(cache=True)
def compute_all_areas(points, triplet_idx):
    n_trips = len(triplet_idx)
    areas = np.empty(n_trips)
    for t in range(n_trips):
        i, j, k = triplet_idx[t, 0], triplet_idx[t, 1], triplet_idx[t, 2]
        areas[t] = 0.5 * abs(
            points[i, 0] * (points[j, 1] - points[k, 1]) +
            points[j, 0] * (points[k, 1] - points[i, 1]) +
            points[k, 0] * (points[i, 1] - points[j, 1])
        )
    return areas


@njit(cache=True) 
def sa_incremental(points_init, triplet_idx, pt_map, pt_count, seed, max_iter,
                   T_start, T_end, step_start, step_end, focused_prob=0.4):
    """SA with incremental area updates and focused moves."""
    np.random.seed(seed)
    points = points_init.copy()
    n = len(points)
    
    # Compute initial areas
    areas = compute_all_areas(points, triplet_idx)
    current_min = np.min(areas)
    best_min = current_min
    best_points = points.copy()
    
    log_T_s = np.log(T_start)
    log_T_e = np.log(T_end)
    log_st_s = np.log(step_start)
    log_st_e = np.log(step_end)
    
    # Cache min-triangle point indices (refresh periodically)
    min_t_idx = np.argmin(areas)
    min_tri_pts = np.array([triplet_idx[min_t_idx, 0], 
                            triplet_idx[min_t_idx, 1],
                            triplet_idx[min_t_idx, 2]])
    
    for it in range(max_iter):
        frac = float(it) / float(max_iter)
        T = np.exp(log_T_s + frac * (log_T_e - log_T_s))
        step = np.exp(log_st_s + frac * (log_st_e - log_st_s))
        
        # Pick point to move
        if np.random.random() < focused_prob:
            r = np.random.randint(0, 3)
            idx = min_tri_pts[r]
        else:
            idx = np.random.randint(0, n)
        
        old_x = points[idx, 0]
        old_y = points[idx, 1]
        
        dx = np.random.randn() * step
        dy = np.random.randn() * step
        new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)
        
        points[idx, 0] = new_x
        points[idx, 1] = new_y
        
        # Incrementally update areas for triplets involving idx
        n_affected = pt_count[idx]
        old_areas = np.empty(n_affected)
        new_min = 1e10
        
        for ti in range(n_affected):
            t = pt_map[idx, ti]
            old_areas[ti] = areas[t]
            i, j, k = triplet_idx[t, 0], triplet_idx[t, 1], triplet_idx[t, 2]
            new_area = 0.5 * abs(
                points[i, 0] * (points[j, 1] - points[k, 1]) +
                points[j, 0] * (points[k, 1] - points[i, 1]) +
                points[k, 0] * (points[i, 1] - points[j, 1])
            )
            areas[t] = new_area
            if new_area < new_min:
                new_min = new_area
        
        # Check non-affected triplets only if affected min is >= current min
        if new_min >= current_min:
            # Need to find actual min over all areas
            actual_min = new_min
            for t in range(len(areas)):
                if areas[t] < actual_min:
                    actual_min = areas[t]
            new_min = actual_min
        
        delta = new_min - current_min
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_points = points.copy()
            # Refresh focused target
            if it % 10000 == 0:
                min_t_idx = np.argmin(areas)
                min_tri_pts[0] = triplet_idx[min_t_idx, 0]
                min_tri_pts[1] = triplet_idx[min_t_idx, 1]
                min_tri_pts[2] = triplet_idx[min_t_idx, 2]
        else:
            # Restore
            points[idx, 0] = old_x
            points[idx, 1] = old_y
            for ti in range(n_affected):
                t = pt_map[idx, ti]
                areas[t] = old_areas[ti]
    
    return best_points, best_min


def symmetrize(points):
    """Force perfect mirror symmetry about x=0.5."""
    n = len(points)
    used = [False] * n
    pairs = []
    
    for i in range(n):
        if used[i]:
            continue
        mirror_x = 1.0 - points[i, 0]
        best_j = -1
        best_dist = float('inf')
        for j in range(n):
            if j == i or used[j]:
                continue
            d = np.sqrt((points[j, 0] - mirror_x)**2 + (points[j, 1] - points[i, 1])**2)
            if d < best_dist:
                best_dist = d
                best_j = j
        
        if best_dist < 0.05 and best_j >= 0:
            used[i] = True
            used[best_j] = True
            pairs.append((i, best_j))
        else:
            used[i] = True
    
    result = points.copy()
    for i, j in pairs:
        avg_y = (points[i, 1] + points[j, 1]) / 2.0
        if points[i, 0] < points[j, 0]:
            left, right = i, j
        else:
            left, right = j, i
        avg_dist = (0.5 - points[left, 0] + points[right, 0] - 0.5) / 2.0
        result[left, 0] = 0.5 - avg_dist
        result[left, 1] = avg_y
        result[right, 0] = 0.5 + avg_dist
        result[right, 1] = avg_y
    
    # Force unpaired point to x=0.5
    for i in range(n):
        paired = False
        for a, b in pairs:
            if i == a or i == b:
                paired = True
                break
        if not paired:
            result[i, 0] = 0.5
    
    return result


def run_seed(seed):
    """Run optimization for one seed."""
    t0 = time.time()
    
    triplet_idx = TRIPLET_IDX
    pt_map, pt_count = build_point_triplet_map(triplet_idx, 11)
    
    best_pts = REFINE_BEST.copy()
    best_m = np.min(compute_all_areas(best_pts, triplet_idx))
    
    # Try both raw and symmetrized starts
    sym_pts = symmetrize(REFINE_BEST)
    
    for label, start in [("raw", REFINE_BEST.copy()), ("sym", sym_pts.copy())]:
        # Phase 1: moderate exploration
        pts, m = sa_incremental(
            start, triplet_idx, pt_map, pt_count,
            seed=seed,
            max_iter=300_000_000,
            T_start=0.002, T_end=1e-9,
            step_start=0.005, step_end=0.000005,
            focused_prob=0.4
        )
        if m > best_m:
            best_m = m
            best_pts = pts.copy()
        
        # Phase 2: ultra-fine from best
        pts2, m2 = sa_incremental(
            pts if m > np.min(compute_all_areas(start, triplet_idx)) else start,
            triplet_idx, pt_map, pt_count,
            seed=seed + 5000,
            max_iter=300_000_000,
            T_start=0.0003, T_end=1e-11,
            step_start=0.001, step_end=0.000001,
            focused_prob=0.5
        )
        if m2 > best_m:
            best_m = m2
            best_pts = pts2.copy()
    
    elapsed = time.time() - t0
    return {'seed': seed, 'normalized': best_m / TRI_AREA, 'points': best_pts, 'time': elapsed}


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print("Ultra-fine incremental SA")
    print(f"Start: {np.min(compute_all_areas(REFINE_BEST, TRIPLET_IDX))/TRI_AREA:.8f}")
    print(f"SOTA:  0.03652989\n")
    
    # Warmup
    print("JIT warmup...")
    pt_map, pt_count = build_point_triplet_map(TRIPLET_IDX, 11)
    _ = sa_incremental(REFINE_BEST.copy(), TRIPLET_IDX, pt_map, pt_count, 0, 1000,
                       0.01, 1e-5, 0.01, 0.001, 0.4)
    print("Done\n")
    
    t0 = time.time()
    with Pool(3) as pool:
        results = pool.map(run_seed, seeds)
    total = time.time() - t0
    
    metrics = [r['normalized'] for r in results]
    
    print(f"Total: {total:.1f}s\n")
    print("| Seed | Metric | Time |")
    print("|------|--------|------|")
    for r in results:
        print(f"| {r['seed']} | {r['normalized']:.8f} | {r['time']:.1f}s |")
    print(f"| Mean | {np.mean(metrics):.8f} +/- {np.std(metrics):.8f} | |")
    
    best = max(results, key=lambda r: r['normalized'])
    print(f"\nBest: {best['normalized']:.8f}")
    print("BEST_POINTS = np.array([")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
