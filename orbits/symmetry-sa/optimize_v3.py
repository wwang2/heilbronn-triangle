"""
Heilbronn Triangle (n=11) - Basin-hopping + soft-min SA.

The known best at 0.03630 is a deep local optimum. To escape:
1. Use soft-min (LSE) objective for smoother landscape
2. Basin-hopping: large random perturbations between SA runs
3. Swap moves: exchange two points
4. Much longer SA with adaptive schedule
"""

import numpy as np
import itertools
import time
from multiprocessing import Pool

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * abs(V0[0]*(V1[1]-V2[1]) + V1[0]*(V2[1]-V0[1]) + V2[0]*(V0[1]-V1[1]))
CENTROID = (V0 + V1 + V2) / 3

TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

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


def all_areas(points):
    a = points[TRIPLET_IDX[:, 0]]
    b = points[TRIPLET_IDX[:, 1]]
    c = points[TRIPLET_IDX[:, 2]]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )


def min_area(points):
    return np.min(all_areas(points))


def soft_min(points, beta=200.0):
    """Smooth approximation of min using log-sum-exp.
    soft_min = -1/beta * log(sum(exp(-beta * areas)))
    This is differentiable and approaches true min as beta -> inf.
    """
    areas = all_areas(points)
    # For numerical stability
    min_a = np.min(areas)
    return min_a - (1.0/beta) * np.log(np.sum(np.exp(-beta * (areas - min_a))))


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


def random_point_in_triangle(rng):
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return V0 + r1 * (V1 - V0) + r2 * (V2 - V0)


def basin_hopping_sa(seed=42, n_hops=30, sa_iters=200_000, time_limit=280):
    """Basin-hopping: alternate between large random perturbations and SA refinement."""
    rng = np.random.RandomState(seed)
    
    best_points = KNOWN_BEST.copy()
    best_min = min_area(best_points)
    
    t0 = time.time()
    
    for hop in range(n_hops):
        if time.time() - t0 > time_limit:
            break
            
        # Generate starting point: large perturbation of best known
        # or completely random with some probability
        if rng.random() < 0.3:
            # Completely random start
            init = np.array([random_point_in_triangle(rng) for _ in range(11)])
        elif rng.random() < 0.5:
            # Large perturbation of known best
            init = KNOWN_BEST.copy()
            # Randomly relocate 2-4 points
            n_relocate = rng.randint(2, 5)
            relocate_idx = rng.choice(11, n_relocate, replace=False)
            for idx in relocate_idx:
                init[idx] = random_point_in_triangle(rng)
        else:
            # Medium perturbation of best found so far
            init = best_points.copy()
            perturb = rng.choice([0.03, 0.05, 0.08, 0.12])
            for i in range(11):
                dx = rng.randn() * perturb
                dy = rng.randn() * perturb
                init[i, 0], init[i, 1] = project_to_triangle(
                    init[i, 0] + dx, init[i, 1] + dy)
        
        # SA refinement
        points = init.copy()
        current_obj = soft_min(points, beta=150.0)
        local_best_min = min_area(points)
        local_best_pts = points.copy()
        
        log_T_start = np.log(0.008)
        log_T_end = np.log(1e-7)
        log_step_start = np.log(0.025)
        log_step_end = np.log(0.0003)
        
        for it in range(sa_iters):
            frac = it / sa_iters
            T = np.exp(log_T_start + frac * (log_T_end - log_T_start))
            step = np.exp(log_step_start + frac * (log_step_end - log_step_start))
            
            # Move type: single point perturbation or swap
            if rng.random() < 0.05:
                # Swap two points
                i, j = rng.choice(11, 2, replace=False)
                points[i], points[j] = points[j].copy(), points[i].copy()
                new_obj = soft_min(points, beta=150.0)
                new_real_min = min_area(points)
                
                delta = new_obj - current_obj
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_obj = new_obj
                    if new_real_min > local_best_min:
                        local_best_min = new_real_min
                        local_best_pts = points.copy()
                else:
                    points[i], points[j] = points[j].copy(), points[i].copy()
            else:
                idx = rng.randint(11)
                old_x, old_y = points[idx].copy()
                dx = rng.randn() * step
                dy = rng.randn() * step
                new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)
                points[idx] = [new_x, new_y]
                
                new_obj = soft_min(points, beta=150.0)
                new_real_min = min_area(points)
                
                delta = new_obj - current_obj
                if delta > 0 or rng.random() < np.exp(delta / T):
                    current_obj = new_obj
                    if new_real_min > local_best_min:
                        local_best_min = new_real_min
                        local_best_pts = points.copy()
                else:
                    points[idx] = [old_x, old_y]
        
        # Final polish with tight SA on real min
        points = local_best_pts.copy()
        current_min = local_best_min
        
        for it in range(100_000):
            frac = it / 100_000
            T = np.exp(np.log(0.001) + frac * (np.log(1e-9) - np.log(0.001)))
            step = np.exp(np.log(0.005) + frac * (np.log(0.00005) - np.log(0.005)))
            
            idx = rng.randint(11)
            old_x, old_y = points[idx].copy()
            dx = rng.randn() * step
            dy = rng.randn() * step
            new_x, new_y = project_to_triangle(old_x + dx, old_y + dy)
            points[idx] = [new_x, new_y]
            new_min = min_area(points)
            
            delta = new_min - current_min
            if delta > 0 or rng.random() < np.exp(delta / T):
                current_min = new_min
                if new_min > local_best_min:
                    local_best_min = new_min
                    local_best_pts = points.copy()
            else:
                points[idx] = [old_x, old_y]
        
        if local_best_min > best_min:
            best_min = local_best_min
            best_points = local_best_pts.copy()
            print(f"  Hop {hop}: NEW BEST {best_min/TRI_AREA:.6f}")
    
    return best_points, best_min


def run_seed(seed):
    """Run basin-hopping for one seed."""
    t0 = time.time()
    pts, m = basin_hopping_sa(seed=seed, n_hops=40, sa_iters=150_000, time_limit=280)
    elapsed = time.time() - t0
    norm = m / TRI_AREA
    return {'seed': seed, 'points': pts, 'normalized': norm, 'time': elapsed}


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print(f"Basin-hopping SA with {len(seeds)} seeds")
    print(f"Known best normalized: {min_area(KNOWN_BEST)/TRI_AREA:.6f}")
    print()

    t0 = time.time()

    with Pool(3) as pool:
        results = pool.map(run_seed, seeds)

    total_time = time.time() - t0

    metrics = [r['normalized'] for r in results]
    mean_m = np.mean(metrics)
    std_m = np.std(metrics)

    print(f"\nTotal time: {total_time:.1f}s\n")
    print(f"| Seed | Metric | Time |")
    print(f"|------|--------|------|")
    for r in results:
        print(f"| {r['seed']} | {r['normalized']:.6f} | {r['time']:.1f}s |")
    print(f"| **Mean** | **{mean_m:.6f} +/- {std_m:.6f}** | |")

    best_result = max(results, key=lambda r: r['normalized'])
    print(f"\nBest normalized: {best_result['normalized']:.6f}")
    print("BEST_POINTS = np.array([")
    for p in best_result['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
