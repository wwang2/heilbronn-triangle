"""
Coordinate descent / pattern search for ultra-fine refinement.

Systematically try perturbations along each axis for each point.
If min_area improves, accept. Halve step size when no improvement found.
"""

import numpy as np
import itertools
import time

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

def min_area(points):
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return np.min(areas)

def coordinate_descent(points, min_step=1e-14, max_rounds=200):
    """Hooke-Jeeves pattern search."""
    pts = points.copy()
    best_m = min_area(pts)
    step = 0.001
    
    round_num = 0
    while step > min_step and round_num < max_rounds:
        improved = False
        for i in range(11):
            for dim in range(2):
                # Try +step
                old_val = pts[i, dim]
                pts[i, dim] = old_val + step
                pts[i, 0], pts[i, 1] = project_to_triangle(pts[i, 0], pts[i, 1])
                m = min_area(pts)
                if m > best_m:
                    best_m = m
                    improved = True
                    continue
                pts[i, 0], pts[i, 1] = points[i, 0], points[i, 1]  # restore
                pts[i] = points[i].copy()  # full restore
                
                # Actually need to be more careful with restore
                pts = points.copy()  # brute force restore
                pts[i, dim] = old_val + step
                pts[i, 0], pts[i, 1] = project_to_triangle(pts[i, 0], pts[i, 1])
                m = min_area(pts)
                if m > best_m:
                    best_m = m
                    points = pts.copy()
                    improved = True
                    continue
                
                # Try -step
                pts = points.copy()
                pts[i, dim] = old_val - step
                pts[i, 0], pts[i, 1] = project_to_triangle(pts[i, 0], pts[i, 1])
                m = min_area(pts)
                if m > best_m:
                    best_m = m
                    points = pts.copy()
                    improved = True
                    continue
                
                pts = points.copy()
        
        if not improved:
            step /= 2.0
        round_num += 1
        if round_num % 10 == 0:
            print(f"  Round {round_num}: step={step:.2e}, metric={best_m/TRI_AREA:.10f}")
    
    return points, best_m

if __name__ == '__main__':
    print("Coordinate descent refinement")
    print(f"Start: {min_area(REFINE_BEST)/TRI_AREA:.10f}")
    
    t0 = time.time()
    pts, m = coordinate_descent(REFINE_BEST.copy())
    elapsed = time.time() - t0
    
    print(f"\nFinal: {m/TRI_AREA:.10f} (time: {elapsed:.1f}s)")
    print(f"Improvement: {(m/TRI_AREA - min_area(REFINE_BEST)/TRI_AREA):.2e}")
    
    print("\nBEST_POINTS = np.array([")
    for p in pts:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
