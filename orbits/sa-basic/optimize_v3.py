"""
SA optimization v3: incremental area updates (only recompute triplets involving
the moved point), much faster per iteration.
"""

import numpy as np
import time
import sys
import os
import itertools
from multiprocessing import Pool

EQUILATERAL_AREA = 0.5 * np.sqrt(3) / 2
N_POINTS = 11
SQRT3 = np.sqrt(3)

# Precompute all triplet indices and per-point triplet membership
ALL_TRIPLETS = np.array(list(itertools.combinations(range(N_POINTS), 3)))  # (165, 3)
N_TRIPLETS = len(ALL_TRIPLETS)

# For each point i, which triplet indices contain it?
POINT_TRIPLETS = []
for i in range(N_POINTS):
    mask = np.any(ALL_TRIPLETS == i, axis=1)
    POINT_TRIPLETS.append(np.where(mask)[0])


def compute_all_areas(points):
    """Compute all 165 triangle areas vectorized."""
    a = points[ALL_TRIPLETS[:, 0]]
    b = points[ALL_TRIPLETS[:, 1]]
    c = points[ALL_TRIPLETS[:, 2]]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )


def update_areas(areas, points, moved_idx):
    """Update only the ~45 triplets containing moved_idx."""
    tri_ids = POINT_TRIPLETS[moved_idx]
    a = points[ALL_TRIPLETS[tri_ids, 0]]
    b = points[ALL_TRIPLETS[tri_ids, 1]]
    c = points[ALL_TRIPLETS[tri_ids, 2]]
    new_areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    areas[tri_ids] = new_areas
    return areas


def random_point_in_triangle(rng):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, SQRT3 / 2])
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return v0 + r1 * (v1 - v0) + r2 * (v2 - v0)


def project_to_triangle(x, y):
    y = max(y, 0.0)
    if y > SQRT3 * x:
        t = (x + SQRT3 * y) / 4.0
        t = max(0.0, min(0.5, t))
        x, y = t, SQRT3 * t
    if y > SQRT3 * (1 - x):
        t = (x - y / SQRT3 + 1) / 2.0
        t = max(0.5, min(1.0, t))
        x, y = t, SQRT3 * (1 - t)
    y = max(y, 0.0)
    return x, y


def sa_fast(seed, max_iter=1000000, n_restarts=30):
    """Fast SA with incremental area updates."""
    overall_best_obj = -1
    overall_best_pts = None
    
    for restart in range(n_restarts):
        rng = np.random.RandomState(seed + restart * 1000)
        points = np.array([random_point_in_triangle(rng) for _ in range(N_POINTS)])
        
        areas = compute_all_areas(points)
        current_min = float(np.min(areas))
        best_obj = current_min
        best_points = points.copy()
        
        T_start = 0.015
        T_end = 1e-9
        cooling = (T_end / T_start) ** (1.0 / max_iter)
        
        p_start = 0.08
        p_end = 0.0005
        p_rate = (p_end / p_start) ** (1.0 / max_iter)
        
        T = T_start
        ps = p_start
        no_improve = 0
        
        for it in range(max_iter):
            idx = rng.randint(N_POINTS)
            old_point = points[idx].copy()
            old_areas = areas[POINT_TRIPLETS[idx]].copy()
            
            dx = rng.randn() * ps
            dy = rng.randn() * ps
            nx, ny = project_to_triangle(old_point[0] + dx, old_point[1] + dy)
            points[idx] = [nx, ny]
            
            areas = update_areas(areas, points, idx)
            new_min = float(np.min(areas))
            
            delta = new_min - current_min
            
            if delta > 0:
                current_min = new_min
                no_improve = 0
                if new_min > best_obj:
                    best_obj = new_min
                    best_points = points.copy()
            elif T > 0 and rng.random() < np.exp(delta / T):
                current_min = new_min
                no_improve += 1
            else:
                points[idx] = old_point
                areas[POINT_TRIPLETS[idx]] = old_areas
                no_improve += 1
            
            T *= cooling
            ps *= p_rate
            
            if no_improve > 5000:
                T = max(T, T_start * 0.03)
                ps = max(ps, p_start * 0.15)
                no_improve = 0
        
        # Local refinement
        points = best_points.copy()
        areas = compute_all_areas(points)
        current_min = best_obj
        
        for ri in range(100000):
            idx = rng.randint(N_POINTS)
            old_point = points[idx].copy()
            old_areas_r = areas[POINT_TRIPLETS[idx]].copy()
            
            scale = 0.001 * (1 - ri / 100000) + 0.0001
            dx = rng.randn() * scale
            dy = rng.randn() * scale
            nx, ny = project_to_triangle(old_point[0] + dx, old_point[1] + dy)
            points[idx] = [nx, ny]
            areas = update_areas(areas, points, idx)
            new_min = float(np.min(areas))
            
            if new_min > current_min:
                current_min = new_min
                if new_min > best_obj:
                    best_obj = new_min
                    best_points = points.copy()
            else:
                points[idx] = old_point
                areas[POINT_TRIPLETS[idx]] = old_areas_r
        
        if best_obj > overall_best_obj:
            overall_best_obj = best_obj
            overall_best_pts = best_points.copy()
    
    return overall_best_obj, overall_best_pts


def run_one_seed(seed):
    t0 = time.time()
    obj, pts = sa_fast(seed, max_iter=1000000, n_restarts=30)
    elapsed = time.time() - t0
    normalized = obj / EQUILATERAL_AREA
    return {'seed': seed, 'min_area_normalized': normalized, 'points': pts, 'time': elapsed}


def format_points_array(points):
    lines = ["np.array(["]
    for i, (x, y) in enumerate(points):
        comma = "," if i < len(points) - 1 else ""
        lines.append(f"    [{x:.15f}, {y:.15f}]{comma}")
    lines.append("])")
    return "\n".join(lines)


def update_solution_file(points, metric):
    solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
    with open(solution_path, 'r') as f:
        content = f.read()
    
    points_str = format_points_array(points)
    new_assignment = f"BEST_POINTS = {points_str}"
    
    import re
    if "BEST_POINTS = None" in content:
        content = content.replace("BEST_POINTS = None", new_assignment)
    else:
        pattern = r'BEST_POINTS = np\.array\(\[.*?\]\)'
        content = re.sub(pattern, new_assignment, content, flags=re.DOTALL)
    
    with open(solution_path, 'w') as f:
        f.write(content)
    print(f"Updated solution.py with best points (metric={metric:.6f})")


if __name__ == '__main__':
    seeds = [42, 123, 7]
    n_workers = min(len(seeds), os.cpu_count() or 4)
    
    print(f"Running SA v3 (incremental updates): {len(seeds)} seeds, {n_workers} workers")
    print(f"Each seed: 30 restarts x 1M SA iters + 100k refinement")
    
    t_total = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_one_seed, seeds)
    t_total = time.time() - t_total
    
    print(f"\n{'Seed':>8} | {'Metric':>10} | {'Time (s)':>8}")
    print("-" * 35)
    metrics = []
    for r in results:
        print(f"{r['seed']:>8} | {r['min_area_normalized']:>10.6f} | {r['time']:>8.1f}")
        metrics.append(r['min_area_normalized'])
    
    mean_m = np.mean(metrics)
    std_m = np.std(metrics)
    print("-" * 35)
    print(f"{'Mean':>8} | {mean_m:>10.6f} +/- {std_m:.6f}")
    print(f"Total wall time: {t_total:.1f}s")
    
    best = max(results, key=lambda r: r['min_area_normalized'])
    print(f"\nBest seed={best['seed']}, metric={best['min_area_normalized']:.6f}")
    print(f"Combined score: {best['min_area_normalized'] / 0.036530:.4f}")
    
    update_solution_file(best['points'], best['min_area_normalized'])
