"""
SA optimization v2: improved cooling, larger restarts, local refinement phase.
"""

import numpy as np
import time
import sys
import os
import itertools
from multiprocessing import Pool

EQUILATERAL_AREA = 0.5 * np.sqrt(3) / 2
N_POINTS = 11


def triangle_area_all(points):
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return areas


def min_triangle_area(points):
    return float(np.min(triangle_area_all(points)))


def random_point_in_triangle(rng):
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return v0 + r1 * (v1 - v0) + r2 * (v2 - v0)


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


def sa_with_refinement(seed, max_iter=500000, n_restarts=20):
    """SA with multiple restarts + greedy local refinement."""
    rng = np.random.RandomState(seed)
    
    overall_best_obj = -1
    overall_best_pts = None
    
    for restart in range(n_restarts):
        rseed = seed + restart * 1000
        rng_local = np.random.RandomState(rseed)
        
        points = np.array([random_point_in_triangle(rng_local) for _ in range(N_POINTS)])
        current_obj = min_triangle_area(points)
        best_obj = current_obj
        best_points = points.copy()
        
        # Adaptive temperature: start higher, cool slower
        T_start = 0.02
        T_end = 1e-8
        cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
        
        perturb_start = 0.1
        perturb_end = 0.001
        scale_rate = (perturb_end / perturb_start) ** (1.0 / max_iter)
        
        T = T_start
        perturb_scale = perturb_start
        no_improve = 0
        
        for it in range(max_iter):
            idx = rng_local.randint(N_POINTS)
            old_point = points[idx].copy()
            
            dx = rng_local.randn() * perturb_scale
            dy = rng_local.randn() * perturb_scale
            new_x, new_y = project_to_triangle(old_point[0] + dx, old_point[1] + dy)
            points[idx] = [new_x, new_y]
            
            new_obj = min_triangle_area(points)
            delta = new_obj - current_obj
            
            if delta > 0:
                current_obj = new_obj
                no_improve = 0
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_points = points.copy()
            elif T > 0 and rng_local.random() < np.exp(delta / T):
                current_obj = new_obj
                no_improve += 1
            else:
                points[idx] = old_point
                no_improve += 1
            
            T *= cooling_rate
            perturb_scale *= scale_rate
            
            # Reheat with decay
            if no_improve > 3000:
                T = max(T, T_start * 0.05)
                perturb_scale = max(perturb_scale, perturb_start * 0.2)
                no_improve = 0
        
        # Local refinement: greedy hill-climbing with tiny perturbations
        points = best_points.copy()
        current_obj = best_obj
        for refine_iter in range(50000):
            idx = rng_local.randint(N_POINTS)
            old_point = points[idx].copy()
            scale = 0.002 * (1 - refine_iter / 50000) + 0.0002
            dx = rng_local.randn() * scale
            dy = rng_local.randn() * scale
            new_x, new_y = project_to_triangle(old_point[0] + dx, old_point[1] + dy)
            points[idx] = [new_x, new_y]
            new_obj = min_triangle_area(points)
            if new_obj > current_obj:
                current_obj = new_obj
                if new_obj > best_obj:
                    best_obj = new_obj
                    best_points = points.copy()
            else:
                points[idx] = old_point
        
        if best_obj > overall_best_obj:
            overall_best_obj = best_obj
            overall_best_pts = best_points.copy()
    
    return overall_best_obj, overall_best_pts


def run_one_seed(seed):
    t0 = time.time()
    obj, pts = sa_with_refinement(seed, max_iter=500000, n_restarts=20)
    elapsed = time.time() - t0
    normalized = obj / EQUILATERAL_AREA
    return {
        'seed': seed,
        'min_area_normalized': normalized,
        'points': pts,
        'time': elapsed,
    }


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
    
    print(f"Running SA v2 optimization: {len(seeds)} seeds, {n_workers} workers")
    print(f"Each seed: 20 restarts x 500k SA iterations + 50k refinement")
    
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
    
    # Only update if better than current
    # Read current solution to check
    current_metric = 0.032625  # from v1
    if best['min_area_normalized'] > current_metric:
        update_solution_file(best['points'], best['min_area_normalized'])
    else:
        print(f"No improvement over current {current_metric:.6f}, keeping existing solution")
