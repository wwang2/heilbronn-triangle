"""
Run SA optimization with multiple seeds in parallel, find best coordinates,
then update solution.py with hardcoded best points.
"""

import numpy as np
import time
import sys
import os
from multiprocessing import Pool

# Add the orbit directory to path so we can import solution
sys.path.insert(0, os.path.dirname(__file__))
from solution import simulated_annealing, min_triangle_area

EQUILATERAL_AREA = 0.5 * np.sqrt(3) / 2


def run_one_seed(args):
    """Run SA with given seed and parameters, return results."""
    seed, max_iter, n_restarts = args
    t0 = time.time()
    
    best_obj = -1
    best_pts = None
    
    for restart in range(n_restarts):
        obj, pts = simulated_annealing(
            max_iter=max_iter,
            seed=seed + restart * 1000,
            T_start=0.05,
            T_end=1e-7,
            perturb_scale_start=0.12,
            perturb_scale_end=0.002
        )
        if obj > best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    elapsed = time.time() - t0
    normalized = best_obj / EQUILATERAL_AREA
    
    return {
        'seed': seed,
        'min_area': best_obj,
        'min_area_normalized': normalized,
        'points': best_pts,
        'time': elapsed,
    }


def format_points_array(points):
    """Format points as a numpy array literal for hardcoding."""
    lines = ["np.array(["]
    for i, (x, y) in enumerate(points):
        comma = "," if i < len(points) - 1 else ""
        lines.append(f"    [{x:.15f}, {y:.15f}]{comma}")
    lines.append("])")
    return "\n".join(lines)


def update_solution_file(points, metric):
    """Update solution.py to hardcode the best points."""
    solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
    
    with open(solution_path, 'r') as f:
        content = f.read()
    
    # Replace BEST_POINTS = None with the actual array
    points_str = format_points_array(points)
    new_assignment = f"BEST_POINTS = {points_str}"
    
    # Replace the line
    if "BEST_POINTS = None" in content:
        content = content.replace("BEST_POINTS = None", new_assignment)
    elif "BEST_POINTS = np.array" in content:
        # Already has hardcoded points, replace the block
        import re
        pattern = r'BEST_POINTS = np\.array\(\[.*?\]\)'
        content = re.sub(pattern, new_assignment, content, flags=re.DOTALL)
    
    with open(solution_path, 'w') as f:
        f.write(content)
    
    print(f"Updated solution.py with best points (metric={metric:.6f})")


if __name__ == '__main__':
    seeds = [42, 123, 7, 999, 2024, 314, 271, 1618]
    n_restarts = 15
    max_iter = 300000
    
    print(f"Running SA optimization: {len(seeds)} seeds x {n_restarts} restarts x {max_iter} iterations")
    print(f"Using multiprocessing with {min(len(seeds), os.cpu_count() or 4)} workers")
    
    args_list = [(s, max_iter, n_restarts) for s in seeds]
    
    t_total = time.time()
    with Pool(min(len(seeds), os.cpu_count() or 4)) as pool:
        results = pool.map(run_one_seed, args_list)
    t_total = time.time() - t_total
    
    # Print results table
    print(f"\n{'Seed':>8} | {'Metric':>10} | {'Time (s)':>8}")
    print("-" * 35)
    
    metrics = []
    for r in results:
        print(f"{r['seed']:>8} | {r['min_area_normalized']:>10.6f} | {r['time']:>8.1f}")
        metrics.append(r['min_area_normalized'])
    
    mean_metric = np.mean(metrics)
    std_metric = np.std(metrics)
    print("-" * 35)
    print(f"{'Mean':>8} | {mean_metric:>10.6f} ± {std_metric:.6f}")
    print(f"Total wall time: {t_total:.1f}s")
    
    # Find the best overall
    best_result = max(results, key=lambda r: r['min_area_normalized'])
    print(f"\nBest seed: {best_result['seed']}, metric: {best_result['min_area_normalized']:.6f}")
    print(f"Combined score (vs SOTA 0.03653): {best_result['min_area_normalized'] / 0.036530:.4f}")
    
    # Update solution.py with best points
    update_solution_file(best_result['points'], best_result['min_area_normalized'])
    
    # Save detailed results
    np.savez(
        os.path.join(os.path.dirname(__file__), 'optimization_results.npz'),
        best_points=best_result['points'],
        best_metric=best_result['min_area_normalized'],
        all_metrics=np.array(metrics),
        all_seeds=np.array(seeds),
    )
    print("Saved optimization_results.npz")
