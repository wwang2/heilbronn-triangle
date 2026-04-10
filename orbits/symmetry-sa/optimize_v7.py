"""
Start from symmetrized config, run ultra-fine Numba SA.
Also try: symmetric SA where perturbations preserve symmetry.
"""
import numpy as np
import itertools
import time
from numba import njit
from multiprocessing import Pool

TRI_AREA = np.sqrt(3) / 4
TRIPLET_IDX = np.array(list(itertools.combinations(range(11), 3)), dtype=np.int32)

# Symmetrized best
SYM_BEST = np.array([
    [0.1062306703654052, 0.0710761657400959],
    [0.8521750796907640, 0.2560402760808697],
    [0.5000000000000000, 0.2111254344044703],
    [0.2774515797949464, 0.0000000000000000],
    [0.1478249203092360, 0.2560402760808697],
    [0.4279840016586173, 0.7412900423154831],
    [0.5720159983413827, 0.7412900423154831],
    [0.8937693296345948, 0.0710761657400959],
    [0.4093343198961193, 0.4392897750602333],
    [0.7225484202050536, 0.0000000000000000],
    [0.5906656801038807, 0.4392897750602333],
])

# Pairs: (0,7), (1,4), (3,9), (5,6), (8,10). Axis: 2
PAIR_LEFT = np.array([0, 4, 3, 5, 8], dtype=np.int32)  # x < 0.5
PAIR_RIGHT = np.array([7, 1, 9, 6, 10], dtype=np.int32)  # x > 0.5
AXIS_IDX = 2


@njit(cache=True)
def project_to_triangle(x, y):
    sqrt3 = 1.7320508075688772
    if y < 0.0: y = 0.0
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        if t < 0.0: t = 0.0
        elif t > 0.5: t = 0.5
        x = t; y = sqrt3 * t
    if y > sqrt3 * (1.0 - x):
        t = (x - y / sqrt3 + 1.0) / 2.0
        if t < 0.5: t = 0.5
        elif t > 1.0: t = 1.0
        x = t; y = sqrt3 * (1.0 - t)
    if y < 0.0: y = 0.0
    return x, y


@njit(cache=True)
def compute_min_area(points, triplet_idx):
    min_a = 1e10
    for t in range(len(triplet_idx)):
        i, j, k = triplet_idx[t, 0], triplet_idx[t, 1], triplet_idx[t, 2]
        area = 0.5 * abs(
            points[i, 0] * (points[j, 1] - points[k, 1]) +
            points[j, 0] * (points[k, 1] - points[i, 1]) +
            points[k, 0] * (points[i, 1] - points[j, 1])
        )
        if area < min_a:
            min_a = area
    return min_a


@njit(cache=True)
def sa_symmetric_enforced(points_init, triplet_idx, pair_left, pair_right,
                           axis_idx, seed, max_iter,
                           T_start, T_end, step_start, step_end):
    """SA preserving exact mirror symmetry.
    Free variables: 5 (x_left, y) pairs + 1 axis y = 11 params.
    When we move a left point, its mirror moves correspondingly."""
    np.random.seed(seed)
    points = points_init.copy()
    n_pairs = len(pair_left)
    n_free = n_pairs + 1  # 5 pairs + 1 axis point
    
    current_min = compute_min_area(points, triplet_idx)
    best_min = current_min
    best_points = points.copy()
    
    log_T_s = np.log(T_start)
    log_T_e = np.log(T_end)
    log_st_s = np.log(step_start)
    log_st_e = np.log(step_end)
    
    for it in range(max_iter):
        frac = float(it) / float(max_iter)
        T = np.exp(log_T_s + frac * (log_T_e - log_T_s))
        step = np.exp(log_st_s + frac * (log_st_e - log_st_s))
        
        unit = np.random.randint(0, n_free)
        
        if unit < n_pairs:
            li = pair_left[unit]
            ri = pair_right[unit]
            old_lx = points[li, 0]
            old_ly = points[li, 1]
            old_rx = points[ri, 0]
            old_ry = points[ri, 1]
            
            dx = np.random.randn() * step
            dy = np.random.randn() * step
            
            new_lx, new_ly = project_to_triangle(old_lx + dx, old_ly + dy)
            # Ensure left point stays left
            if new_lx > 0.5:
                new_lx = 1.0 - new_lx
            # Mirror
            new_rx = 1.0 - new_lx
            new_ry = new_ly
            
            # Check mirror is in triangle
            new_rx, new_ry = project_to_triangle(new_rx, new_ry)
            # If mirror was projected, re-symmetrize
            if abs(new_rx - (1.0 - new_lx)) > 1e-12 or abs(new_ry - new_ly) > 1e-12:
                # Average
                avg_dist = ((0.5 - new_lx) + (new_rx - 0.5)) / 2.0
                new_lx = 0.5 - avg_dist
                new_ly = (new_ly + new_ry) / 2.0
                new_rx = 0.5 + avg_dist
                new_ry = new_ly
                new_lx, new_ly = project_to_triangle(new_lx, new_ly)
                new_rx, new_ry = project_to_triangle(new_rx, new_ry)
            
            points[li, 0] = new_lx; points[li, 1] = new_ly
            points[ri, 0] = new_rx; points[ri, 1] = new_ry
            
            new_min = compute_min_area(points, triplet_idx)
            delta = new_min - current_min
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_min = new_min
                if new_min > best_min:
                    best_min = new_min
                    best_points = points.copy()
            else:
                points[li, 0] = old_lx; points[li, 1] = old_ly
                points[ri, 0] = old_rx; points[ri, 1] = old_ry
        else:
            # Axis point: only y moves
            old_y = points[axis_idx, 1]
            dy = np.random.randn() * step
            _, new_y = project_to_triangle(0.5, old_y + dy)
            points[axis_idx, 1] = new_y
            
            new_min = compute_min_area(points, triplet_idx)
            delta = new_min - current_min
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_min = new_min
                if new_min > best_min:
                    best_min = new_min
                    best_points = points.copy()
            else:
                points[axis_idx, 1] = old_y
    
    return best_points, best_min


@njit(cache=True)
def sa_unconstrained(points_init, triplet_idx, seed, max_iter,
                     T_start, T_end, step_start, step_end):
    np.random.seed(seed)
    points = points_init.copy()
    n = len(points)
    current_min = compute_min_area(points, triplet_idx)
    best_min = current_min
    best_points = points.copy()
    log_T_s = np.log(T_start)
    log_T_e = np.log(T_end)
    log_st_s = np.log(step_start)
    log_st_e = np.log(step_end)
    
    for it in range(max_iter):
        frac = float(it) / float(max_iter)
        T = np.exp(log_T_s + frac * (log_T_e - log_T_s))
        step = np.exp(log_st_s + frac * (log_st_e - log_st_s))
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


def run_seed(seed):
    t0 = time.time()
    triplet_idx = TRIPLET_IDX
    pl = PAIR_LEFT
    pr = PAIR_RIGHT
    
    best_pts = SYM_BEST.copy()
    best_m = compute_min_area(best_pts, triplet_idx)
    
    # Phase 1: symmetric SA with moderate exploration
    pts, m = sa_symmetric_enforced(
        SYM_BEST.copy(), triplet_idx, pl, pr, AXIS_IDX,
        seed=seed,
        max_iter=200_000_000,
        T_start=0.003, T_end=1e-9,
        step_start=0.008, step_end=0.00002
    )
    if m > best_m:
        best_m = m; best_pts = pts.copy()
    
    # Phase 2: symmetric SA ultra-fine
    pts2, m2 = sa_symmetric_enforced(
        best_pts.copy(), triplet_idx, pl, pr, AXIS_IDX,
        seed=seed + 1000,
        max_iter=200_000_000,
        T_start=0.0005, T_end=1e-11,
        step_start=0.002, step_end=0.000002
    )
    if m2 > best_m:
        best_m = m2; best_pts = pts2.copy()
    
    # Phase 3: unconstrained refinement (allow symmetry breaking)
    pts3, m3 = sa_unconstrained(
        best_pts.copy(), triplet_idx,
        seed=seed + 2000,
        max_iter=200_000_000,
        T_start=0.0003, T_end=1e-11,
        step_start=0.001, step_end=0.000001
    )
    if m3 > best_m:
        best_m = m3; best_pts = pts3.copy()
    
    elapsed = time.time() - t0
    return {'seed': seed, 'normalized': best_m / TRI_AREA, 'points': best_pts, 'time': elapsed}


if __name__ == '__main__':
    seeds = [42, 123, 7]
    
    print("Symmetric-enforced Numba SA + unconstrained refinement")
    print(f"Start: {compute_min_area(SYM_BEST, TRIPLET_IDX)/TRI_AREA:.10f}")
    print()
    
    # Warmup
    print("JIT warmup...")
    _ = compute_min_area(SYM_BEST.copy(), TRIPLET_IDX)
    _ = sa_symmetric_enforced(SYM_BEST.copy(), TRIPLET_IDX, PAIR_LEFT, PAIR_RIGHT, AXIS_IDX, 0, 100, 0.01, 1e-5, 0.01, 0.001)
    _ = sa_unconstrained(SYM_BEST.copy(), TRIPLET_IDX, 0, 100, 0.01, 1e-5, 0.01, 0.001)
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
        print(f"| {r['seed']} | {r['normalized']:.10f} | {r['time']:.1f}s |")
    print(f"| Mean | {np.mean(metrics):.10f} +/- {np.std(metrics):.10f} | |")
    
    best = max(results, key=lambda r: r['normalized'])
    print(f"\nBest: {best['normalized']:.10f}")
    print("BEST_POINTS = np.array([")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    print("])")
