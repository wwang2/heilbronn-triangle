"""
Aggressive simulated annealing with multiple phases for Heilbronn n=11.
Key improvements:
1. Much more iterations
2. Reheat/restart from best solution
3. Better step size adaptation
4. Focus perturbation on the bottleneck (smallest triangle)
"""
import numpy as np
from itertools import combinations
import multiprocessing as mp
import json
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2
TRI_AREA = H / 2  # area = sqrt(3)/4

# Precompute triplet indices
TRIPLETS = np.array(list(combinations(range(N), 3)))
I0, I1, I2 = TRIPLETS[:, 0], TRIPLETS[:, 1], TRIPLETS[:, 2]

# For each point, which triplets involve it
POINT_TRIPLETS = []
for p in range(N):
    mask = (I0 == p) | (I1 == p) | (I2 == p)
    POINT_TRIPLETS.append(np.where(mask)[0])


def is_inside(x, y, tol=0.0):
    return y >= -tol and SQRT3*x >= y - tol and SQRT3*(1-x) >= y - tol


def project(x, y):
    """Project to inside equilateral triangle."""
    if is_inside(x, y):
        return x, y
    # Clamp via barycentric
    lam3 = y / H
    lam2 = x - 0.5 * lam3
    lam1 = 1.0 - lam3 - lam2
    lam1 = max(0, lam1)
    lam2 = max(0, lam2)
    lam3 = max(0, lam3)
    s = lam1 + lam2 + lam3
    if s < 1e-15:
        return 0.5, H/3
    lam1 /= s; lam2 /= s; lam3 /= s
    return lam2 + 0.5*lam3, H*lam3


def all_areas(points):
    """Compute all 165 triangle areas. Vectorized."""
    a, b, c = points[I0], points[I1], points[I2]
    return 0.5 * np.abs(
        a[:, 0]*(b[:, 1]-c[:, 1]) + 
        b[:, 0]*(c[:, 1]-a[:, 1]) + 
        c[:, 0]*(a[:, 1]-b[:, 1])
    )


def partial_areas(points, idx, areas_cache):
    """Update areas for triplets involving point idx. Returns new areas array."""
    new_areas = areas_cache.copy()
    triplet_indices = POINT_TRIPLETS[idx]
    for ti in triplet_indices:
        i, j, k = TRIPLETS[ti]
        a, b, c = points[i], points[j], points[k]
        new_areas[ti] = 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    return new_areas


def random_inside(rng):
    """Generate random point inside equilateral triangle."""
    while True:
        x = rng.uniform(0, 1)
        y = rng.uniform(0, H)
        if is_inside(x, y):
            return x, y


def sa_optimize(seed, total_iters=3000000):
    """Multi-phase simulated annealing."""
    rng = np.random.RandomState(seed)
    
    # Initialize random
    points = np.zeros((N, 2))
    for i in range(N):
        points[i] = random_inside(rng)
    
    areas = all_areas(points)
    current_min = np.min(areas)
    best_min = current_min
    best_points = points.copy()
    
    # SA parameters
    n_phases = 3
    iters_per_phase = total_iters // n_phases
    
    for phase in range(n_phases):
        if phase > 0:
            # Restart from best, with small perturbation
            points = best_points.copy()
            for i in range(N):
                dx, dy = rng.normal(0, 0.02, 2)
                nx, ny = project(points[i, 0]+dx, points[i, 1]+dy)
                points[i] = [nx, ny]
            areas = all_areas(points)
            current_min = np.min(areas)
        
        T_start = 0.005 if phase == 0 else 0.001
        T_end = 1e-8
        T_ratio = (T_end / T_start) ** (1.0 / iters_per_phase)
        
        T = T_start
        for it in range(iters_per_phase):
            T *= T_ratio
            
            # Adaptive step: larger early, smaller late
            base_step = 0.03 * (T / T_start) ** 0.3 + 0.0005
            
            # With some probability, move the point involved in the smallest triangle
            if rng.random() < 0.3:
                min_idx = np.argmin(areas)
                # Pick one of the 3 points in the smallest triangle
                tri = TRIPLETS[min_idx]
                idx = tri[rng.randint(3)]
                step = base_step * 2  # larger step for bottleneck point
            else:
                idx = rng.randint(N)
                step = base_step
            
            old_pos = points[idx].copy()
            dx, dy = rng.normal(0, step, 2)
            nx, ny = project(old_pos[0]+dx, old_pos[1]+dy)
            points[idx] = [nx, ny]
            
            new_areas = partial_areas(points, idx, areas)
            new_min = np.min(new_areas)
            
            delta = new_min - current_min
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-30)):
                areas = new_areas
                current_min = new_min
                if current_min > best_min:
                    best_min = current_min
                    best_points = points.copy()
            else:
                points[idx] = old_pos
    
    return best_min / TRI_AREA, best_points


def run_seed(seed):
    t0 = time.time()
    metric, points = sa_optimize(seed)
    return {'seed': seed, 'metric': metric, 'points': points.tolist(), 'time': time.time()-t0}


def main():
    seeds = list(range(20))
    print(f"Running {len(seeds)} aggressive SA optimizations...")
    t0 = time.time()
    
    with mp.Pool(min(mp.cpu_count(), len(seeds))) as pool:
        results = pool.map(run_seed, seeds)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"\nTop 10:")
    for r in results[:10]:
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  time={r['time']:.1f}s")
    
    best = results[0]
    print(f"\nBest: metric={best['metric']:.10f}")
    print("Points:")
    for x, y in best['points']:
        print(f"  [{x:.16f}, {y:.16f}],")
    
    with open('orbits/known-literature/best_config.json', 'w') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    main()
