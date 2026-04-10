"""
Symmetry-aware Numba-JIT SA for Heilbronn Triangle (n=11).

Key insight from refine-intensive orbit: the optimal config has mirror symmetry
about x=0.5. With 11 points:
- Points come in symmetric pairs (x, y) <-> (1-x, y)  
- One point sits on the axis x=0.5

This means we only need to optimize 6 points (5 pairs + 1 on axis = 11 points,
but only 6 independent positions = 12 free parameters instead of 22).

Strategy:
1. Start from refine-intensive best (0.036530)
2. Enforce mirror symmetry: optimize 6 "half" points, mirror to get all 11
3. Numba JIT for ~100x speedup
4. Multi-phase cooling with reheating
5. Also try unconstrained SA starting from symmetric config (break symmetry)
"""

import numpy as np
import itertools
import time
from numba import njit
from multiprocessing import Pool

TRI_AREA = np.sqrt(3) / 4  # Area of equilateral triangle with side 1

# Precompute triplet indices
TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

# Refine-intensive best (0.036530)
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

# Known literature best (0.036301)
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


@njit(cache=True)
def project_to_triangle(x, y):
    """Project (x,y) to inside equilateral triangle."""
    sqrt3 = np.sqrt(3)
    if y < 0.0:
        y = 0.0
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        if t < 0.0:
            t = 0.0
        elif t > 0.5:
            t = 0.5
        x = t
        y = sqrt3 * t
    if y > sqrt3 * (1.0 - x):
        t = (x - y / sqrt3 + 1.0) / 2.0
        if t < 0.5:
            t = 0.5
        elif t > 1.0:
            t = 1.0
        x = t
        y = sqrt3 * (1.0 - t)
    if y < 0.0:
        y = 0.0
    return x, y


@njit(cache=True)
def compute_all_areas(points, triplet_idx):
    """Compute all 165 triangle areas."""
    n_trips = len(triplet_idx)
    areas = np.empty(n_trips)
    for t in range(n_trips):
        i = triplet_idx[t, 0]
        j = triplet_idx[t, 1]
        k = triplet_idx[t, 2]
        areas[t] = 0.5 * abs(
            points[i, 0] * (points[j, 1] - points[k, 1]) +
            points[j, 0] * (points[k, 1] - points[i, 1]) +
            points[k, 0] * (points[i, 1] - points[j, 1])
        )
    return areas


@njit(cache=True)
def compute_min_area(points, triplet_idx):
    """Compute minimum triangle area."""
    areas = compute_all_areas(points, triplet_idx)
    return np.min(areas)


@njit(cache=True)
def compute_min_area_incremental(points, triplet_idx, point_idx, old_min, 
                                  point_triplets, n_point_triplets):
    """Incrementally compute min area after moving one point.
    Only recompute triplets involving the moved point."""
    new_min = 1e10
    # Check triplets involving moved point
    for ti in range(n_point_triplets):
        t = point_triplets[ti]
        i = triplet_idx[t, 0]
        j = triplet_idx[t, 1]
        k = triplet_idx[t, 2]
        area = 0.5 * abs(
            points[i, 0] * (points[j, 1] - points[k, 1]) +
            points[j, 0] * (points[k, 1] - points[i, 1]) +
            points[k, 0] * (points[i, 1] - points[j, 1])
        )
        if area < new_min:
            new_min = area
    
    if new_min < old_min:
        return new_min
    
    # Also need to check non-involved triplets (their areas unchanged)
    # Actually we need full min. Use cached areas for non-moved triplets.
    # For simplicity, just compute full min when moved triplets don't dominate
    return compute_min_area(points, triplet_idx)


@njit(cache=True)
def precompute_point_triplets(triplet_idx, n_points):
    """For each point, find which triplets involve it."""
    # Max triplets per point for n=11: C(10,2) = 45
    result = np.empty((n_points, 45), dtype=np.int32)
    counts = np.zeros(n_points, dtype=np.int32)
    
    for t in range(len(triplet_idx)):
        for j in range(3):
            p = triplet_idx[t, j]
            result[p, counts[p]] = t
            counts[p] += 1
    
    return result, counts


@njit(cache=True)
def sa_unconstrained(points_init, triplet_idx, seed, max_iter,
                     T_start, T_end, step_start, step_end,
                     point_trips, point_trip_counts):
    """Unconstrained SA with Numba JIT."""
    np.random.seed(seed)
    points = points_init.copy()
    n = len(points)
    
    current_min = compute_min_area(points, triplet_idx)
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
        
        idx = np.random.randint(0, n)
        old_x = points[idx, 0]
        old_y = points[idx, 1]
        
        dx = np.random.randn() * step
        dy = np.random.randn() * step
        new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)
        
        points[idx, 0] = new_x
        points[idx, 1] = new_y
        
        new_min = compute_min_area(points, triplet_idx)
        
        delta = new_min - current_min
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_points = points.copy()
        else:
            points[idx, 0] = old_x
            points[idx, 1] = old_y
    
    return best_points, best_min


@njit(cache=True)
def sa_symmetric(half_points_init, axis_point_init, triplet_idx, seed, max_iter,
                 T_start, T_end, step_start, step_end):
    """SA enforcing mirror symmetry about x=0.5.
    half_points: shape (5, 2) - the 5 points with x < 0.5
    axis_point: shape (2,) - the point on x=0.5 axis
    Full config = half_points + mirror(half_points) + axis_point = 11 points
    """
    np.random.seed(seed)
    
    half = half_points_init.copy()
    axis = axis_point_init.copy()
    n_half = len(half)
    
    # Build full point set
    points = np.empty((11, 2))
    
    def build_full(half, axis, points):
        for i in range(5):
            points[2*i, 0] = half[i, 0]
            points[2*i, 1] = half[i, 1]
            points[2*i+1, 0] = 1.0 - half[i, 0]
            points[2*i+1, 1] = half[i, 1]
        points[10, 0] = axis[0]
        points[10, 1] = axis[1]
    
    build_full(half, axis, points)
    
    current_min = compute_min_area(points, triplet_idx)
    best_min = current_min
    best_half = half.copy()
    best_axis = axis.copy()
    
    log_T_start = np.log(T_start)
    log_T_end = np.log(T_end)
    log_step_start = np.log(step_start)
    log_step_end = np.log(step_end)
    
    n_free = n_half + 1  # 5 half points + 1 axis point = 6 movable units
    
    for it in range(max_iter):
        frac = it / max_iter
        T = np.exp(log_T_start + frac * (log_T_end - log_T_start))
        step = np.exp(log_step_start + frac * (log_step_end - log_step_start))
        
        unit = np.random.randint(0, n_free)
        
        if unit < n_half:
            # Move a half point (and its mirror)
            old_x = half[unit, 0]
            old_y = half[unit, 1]
            
            dx = np.random.randn() * step
            dy = np.random.randn() * step
            new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)
            
            # Ensure x <= 0.5 (left half)
            if new_x > 0.5:
                new_x = 1.0 - new_x
            
            half[unit, 0] = new_x
            half[unit, 1] = new_y
        else:
            # Move axis point (constrained to x=0.5)
            old_x = axis[0]
            old_y = axis[1]
            
            dy = np.random.randn() * step
            new_x = 0.5
            new_y, _ = project_to_triangle(0.5, old_y + dy)
            new_y = _  # project_to_triangle returns (x, y)
            new_x, new_y = project_to_triangle(0.5, old_y + dy)
            
            axis[0] = new_x
            axis[1] = new_y
        
        build_full(half, axis, points)
        new_min = compute_min_area(points, triplet_idx)
        
        delta = new_min - current_min
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_min = new_min
            if new_min > best_min:
                best_min = new_min
                best_half = half.copy()
                best_axis = axis.copy()
        else:
            if unit < n_half:
                half[unit, 0] = old_x
                half[unit, 1] = old_y
            else:
                axis[0] = old_x
                axis[1] = old_y
    
    # Build final best points
    build_full(best_half, best_axis, points)
    return points, best_min


def extract_symmetric_params(points):
    """Extract half-points and axis point from a (near-)symmetric config.
    Pair up points by matching (x,y) with closest (1-x, y)."""
    n = len(points)
    used = [False] * n
    half_points = []
    axis_point = None
    
    # Find points closest to x=0.5 for axis
    axis_dists = [abs(points[i, 0] - 0.5) for i in range(n)]
    
    # Try to pair points
    pairs = []
    remaining = list(range(n))
    
    for i in remaining:
        if used[i]:
            continue
        mirror_x = 1.0 - points[i, 0]
        mirror_y = points[i, 1]
        
        best_j = -1
        best_dist = float('inf')
        for j in remaining:
            if j == i or used[j]:
                continue
            d = np.sqrt((points[j, 0] - mirror_x)**2 + (points[j, 1] - mirror_y)**2)
            if d < best_dist:
                best_dist = d
                best_j = j
        
        if best_dist < 0.01 and best_j >= 0:
            # Good pair
            used[i] = True
            used[best_j] = True
            # Take the one with x < 0.5
            if points[i, 0] <= 0.5:
                half_points.append(points[i].copy())
            else:
                half_points.append(np.array([1.0 - points[i, 0], points[i, 1]]))
    
    # Remaining point is the axis point
    for i in range(n):
        if not used[i]:
            axis_point = np.array([0.5, points[i, 1]])
            break
    
    return np.array(half_points), axis_point


def run_task(args):
    """Run a single optimization task."""
    task_id, mode, start_config, seed = args
    
    triplet_idx = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)
    point_trips, point_trip_counts = precompute_point_triplets(triplet_idx, 11)
    
    best_pts = start_config.copy()
    best_m = compute_min_area(start_config, triplet_idx)
    
    if mode == 'symmetric':
        half, axis = extract_symmetric_params(start_config)
        if len(half) != 5 or axis is None:
            # Fallback to unconstrained
            mode = 'unconstrained'
        else:
            # Multi-phase symmetric SA
            phases = [
                (100_000_000, 0.005, 1e-7, 0.01, 0.0002),
                (100_000_000, 0.002, 1e-8, 0.005, 0.00005),
                (100_000_000, 0.001, 1e-9, 0.002, 0.00001),
            ]
            for phase_i, (iters, ts, te, ss, se) in enumerate(phases):
                pts, m = sa_symmetric(
                    half, axis, triplet_idx,
                    seed=seed + phase_i * 1000,
                    max_iter=iters,
                    T_start=ts, T_end=te,
                    step_start=ss, step_end=se
                )
                if m > best_m:
                    best_m = m
                    best_pts = pts.copy()
                    half, axis = extract_symmetric_params(pts)
    
    if mode == 'unconstrained':
        # Multi-phase unconstrained SA
        phases = [
            (100_000_000, 0.008, 1e-7, 0.015, 0.0003),
            (100_000_000, 0.003, 1e-8, 0.005, 0.00005),
            (100_000_000, 0.001, 1e-9, 0.002, 0.00001),
        ]
        pts = best_pts.copy()
        for phase_i, (iters, ts, te, ss, se) in enumerate(phases):
            pts, m = sa_unconstrained(
                pts, triplet_idx,
                seed=seed + phase_i * 1000,
                max_iter=iters,
                T_start=ts, T_end=te,
                step_start=ss, step_end=se,
                point_trips=point_trips,
                point_trip_counts=point_trip_counts
            )
            if m > best_m:
                best_m = m
                best_pts = pts.copy()
    
    elif mode == 'sym_then_free':
        # First do symmetric, then unconstrained refinement
        half, axis = extract_symmetric_params(start_config)
        if len(half) == 5 and axis is not None:
            pts, m = sa_symmetric(
                half, axis, triplet_idx,
                seed=seed,
                max_iter=150_000_000,
                T_start=0.005, T_end=1e-8,
                step_start=0.01, step_end=0.0001
            )
            if m > best_m:
                best_m = m
                best_pts = pts.copy()
        
        # Then unconstrained refinement
        pts, m = sa_unconstrained(
            best_pts, triplet_idx,
            seed=seed + 5000,
            max_iter=150_000_000,
            T_start=0.002, T_end=1e-9,
            step_start=0.003, step_end=0.00005,
            point_trips=point_trips,
            point_trip_counts=point_trip_counts
        )
        if m > best_m:
            best_m = m
            best_pts = pts.copy()
    
    return task_id, best_pts, best_m


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print("Symmetry-aware Numba-JIT SA")
    print(f"Refine-intensive best: {compute_min_area(REFINE_BEST, TRIPLET_IDX)/TRI_AREA:.6f}")
    print(f"Known-literature best: {compute_min_area(KNOWN_BEST, TRIPLET_IDX)/TRI_AREA:.6f}")
    print()
    
    # Warm up numba
    print("Warming up Numba JIT...")
    dummy = REFINE_BEST.copy()
    _ = compute_min_area(dummy, TRIPLET_IDX)
    pt, pc = precompute_point_triplets(TRIPLET_IDX, 11)
    _ = sa_unconstrained(dummy, TRIPLET_IDX, 0, 100, 0.01, 1e-5, 0.01, 0.001, pt, pc)
    half, axis = extract_symmetric_params(REFINE_BEST)
    if len(half) == 5 and axis is not None:
        _ = sa_symmetric(half, axis, TRIPLET_IDX, 0, 100, 0.01, 1e-5, 0.01, 0.001)
    print("JIT warm-up done\n")
    
    t0 = time.time()
    
    # Build tasks: for each seed, run symmetric, unconstrained, and hybrid
    all_tasks = []
    for seed in seeds:
        all_tasks.append((f"{seed}_sym", 'symmetric', REFINE_BEST.copy(), seed))
        all_tasks.append((f"{seed}_free", 'unconstrained', REFINE_BEST.copy(), seed + 10000))
        all_tasks.append((f"{seed}_hybrid", 'sym_then_free', REFINE_BEST.copy(), seed + 20000))
        # Also try from known-literature
        all_tasks.append((f"{seed}_known", 'unconstrained', KNOWN_BEST.copy(), seed + 30000))
    
    print(f"Running {len(all_tasks)} tasks across {min(len(all_tasks), 6)} workers...")
    
    with Pool(min(len(all_tasks), 6)) as pool:
        all_results = pool.map(run_task, all_tasks)
    
    total_time = time.time() - t0
    
    # Group by seed
    results = []
    for seed in seeds:
        prefix = str(seed) + "_"
        best_m = -1
        best_pts = None
        best_task = None
        for tid, pts, m in all_results:
            if tid.startswith(prefix) and m > best_m:
                best_m = m
                best_pts = pts.copy()
                best_task = tid
        norm = best_m / TRI_AREA
        results.append({'seed': seed, 'normalized': norm, 'points': best_pts, 'task': best_task})
        print(f"Seed {seed}: {norm:.6f} (best from {best_task})")
    
    metrics = [r['normalized'] for r in results]
    mean_m = np.mean(metrics)
    std_m = np.std(metrics)
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\n| Seed | Metric | Best Task |")
    print(f"|------|--------|-----------|")
    for r in results:
        print(f"| {r['seed']} | {r['normalized']:.6f} | {r['task']} |")
    print(f"| **Mean** | **{mean_m:.6f} +/- {std_m:.6f}** | |")
    
    best_result = max(results, key=lambda r: r['normalized'])
    print(f"\nBest normalized: {best_result['normalized']:.6f}")
    print("BEST_POINTS = np.array([")
    for p in best_result['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
