"""
Simulated annealing optimization for Heilbronn triangle problem, n=11.
Uses direct Cartesian coordinates with projection onto the equilateral triangle.
"""
import numpy as np
from itertools import combinations
import multiprocessing as mp
import json
import time

N = 11
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = 0.5 * np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute all triplet indices
TRIPLETS = np.array(list(combinations(range(N), 3)))
I0, I1, I2 = TRIPLETS[:, 0], TRIPLETS[:, 1], TRIPLETS[:, 2]


def is_inside_triangle(x, y, tol=0.0):
    """Check if point (x,y) is inside the equilateral triangle."""
    return (y >= -tol and 
            SQRT3 * x <= SQRT3 - y + tol and 
            y <= SQRT3 * x + tol)


def project_to_triangle(x, y):
    """Project point to nearest point inside equilateral triangle."""
    # Clamp using barycentric coordinates
    h = SQRT3 / 2
    # Barycentric: lam3 = y/h, then from x equation
    lam3 = y / h
    lam12 = 1.0 - lam3
    if lam12 > 1e-10:
        # x = lam2 + 0.5*lam3 => lam2 = (x - 0.5*lam3)
        # But also lam1 + lam2 = lam12
        lam2 = x - 0.5 * lam3
        lam1 = lam12 - lam2
    else:
        lam1 = 0
        lam2 = 0
        lam3 = 1.0
    
    # Clamp all to [0, 1] and renormalize
    lam1 = max(0, lam1)
    lam2 = max(0, lam2)
    lam3 = max(0, lam3)
    s = lam1 + lam2 + lam3
    if s < 1e-15:
        return 0.5, h/3  # centroid
    lam1 /= s
    lam2 /= s
    lam3 /= s
    
    nx = lam1 * V0[0] + lam2 * V1[0] + lam3 * V2[0]
    ny = lam1 * V0[1] + lam2 * V1[1] + lam3 * V2[1]
    return nx, ny


def min_area_vectorized(points):
    """Compute min triangle area among all triplets. Fully vectorized."""
    a = points[I0]
    b = points[I1]  
    c = points[I2]
    areas = 0.5 * np.abs(a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1]))
    return np.min(areas)


def simulated_annealing(seed, n_iter=500000, T_start=0.1, T_end=1e-6):
    """Run simulated annealing to maximize min triangle area."""
    rng = np.random.RandomState(seed)
    
    # Initialize: random points inside triangle via rejection sampling
    points = np.zeros((N, 2))
    for i in range(N):
        while True:
            x = rng.uniform(0, 1)
            y = rng.uniform(0, SQRT3/2)
            if is_inside_triangle(x, y):
                points[i] = [x, y]
                break
    
    current_min = min_area_vectorized(points)
    best_min = current_min
    best_points = points.copy()
    
    # Temperature schedule
    T_ratio = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start
    
    # Step size adapts with temperature
    accept_count = 0
    
    for it in range(n_iter):
        T = T_start * (T_ratio ** it)
        step_size = 0.05 * (T / T_start) ** 0.5 + 0.001
        
        # Pick a random point to move
        idx = rng.randint(N)
        old_pos = points[idx].copy()
        
        # Propose new position
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        new_x = old_pos[0] + dx
        new_y = old_pos[1] + dy
        
        # Project to triangle
        new_x, new_y = project_to_triangle(new_x, new_y)
        points[idx] = [new_x, new_y]
        
        new_min = min_area_vectorized(points)
        
        # Accept or reject
        delta = new_min - current_min
        if delta > 0 or rng.random() < np.exp(delta / (T + 1e-30)):
            current_min = new_min
            accept_count += 1
            if current_min > best_min:
                best_min = current_min
                best_points = points.copy()
        else:
            points[idx] = old_pos
    
    return best_min / TRI_AREA, best_points


def run_with_seed(seed):
    t0 = time.time()
    metric, points = simulated_annealing(seed, n_iter=1000000)
    elapsed = time.time() - t0
    return {'seed': seed, 'metric': metric, 'points': points.tolist(), 'time': elapsed}


def main():
    seeds = list(range(30))
    print(f"Running {len(seeds)} SA optimizations in parallel...")
    t0 = time.time()
    
    with mp.Pool(min(mp.cpu_count(), len(seeds))) as pool:
        results = pool.map(run_with_seed, seeds)
    
    total = time.time() - t0
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print(f"\nTotal time: {total:.1f}s")
    print(f"\nTop 10 results:")
    for r in results[:10]:
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  time={r['time']:.1f}s")
    
    best = results[0]
    print(f"\nBest: seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"Points:")
    for x, y in best['points']:
        print(f"  [{x:.16f}, {y:.16f}],")
    
    with open('orbits/known-literature/best_config.json', 'w') as f:
        json.dump({'metric': best['metric'], 'points': best['points'],
                   'seed': best['seed']}, f, indent=2)
    
    return best


if __name__ == '__main__':
    main()
