"""
Round 3: Iterated targeted Nelder-Mead starting from best round 2 result.
Also try: chained NM (optimize bottleneck, update, repeat many times).
"""
import numpy as np
from itertools import combinations
from multiprocessing import Pool
import time
import json
import os

SQRT3 = np.sqrt(3.0)
TRI_AREA = 0.5 * SQRT3 / 2.0
TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=np.int32)

# Best from round 2 (targeted-nm seed=999, metric=0.0363197608)
BEST_R2 = np.array([
    [0.8560570091098645, 0.0000000000000000],
    [0.6449915126570442, 0.6148927371961720],
    [0.2901914160324213, 0.0000000000000000],
    [0.4292240161835967, 0.3296052441932574],
    [0.5759454920035292, 0.1352913201389639],
    [0.5074482882134488, 0.7412854236706883],
    [0.9262915671662267, 0.1276649516220629],
    [0.3640014747843533, 0.6304690483565014],
    [0.1117348434492683, 0.0555980680459750],
    [0.6736545513847296, 0.2976357798299709],
    [0.1354802865867832, 0.2346587397923007],
])

# Also keep parent as alternative start
PARENT_POINTS = np.array([
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


def compute_all_areas(points):
    p = points[TRIPLETS]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )

def evaluate_config(points):
    return np.min(compute_all_areas(points)) / TRI_AREA

def project_to_triangle(points):
    pts = points.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(y, 0.0)
        if y > SQRT3 * x:
            xp = (x + y / SQRT3) / 2.0; yp = SQRT3 * xp; x, y = xp, yp
        if y > SQRT3 * (1 - x):
            xp = (x - y / SQRT3 + 1) / 2.0; yp = SQRT3 * (1 - xp); x, y = xp, yp
        y = max(y, 0.0)
        pts[i] = [x, y]
    return pts

def check_feasibility(points, tol=1e-9):
    for x, y in points:
        if y < -tol or y > SQRT3 * x + tol or y > SQRT3 * (1 - x) + tol:
            return False
    return True

def find_bottleneck_points(points, k=5):
    areas = compute_all_areas(points)
    idx = np.argsort(areas)[:k]
    point_set = set()
    for i in idx:
        for j in TRIPLETS[i]:
            point_set.add(int(j))
    return list(point_set)

def nelder_mead_subspace(pts, point_indices, scale, max_iter=50000):
    """Run NM on a subspace of point coords."""
    n_vars = len(point_indices) * 2
    
    def objective(x_sub):
        trial = pts.copy()
        for k_idx, pi in enumerate(point_indices):
            trial[pi] = [x_sub[2*k_idx], x_sub[2*k_idx+1]]
        trial = project_to_triangle(trial)
        return -evaluate_config(trial)
    
    x0 = np.array([pts[pi, j] for pi in point_indices for j in range(2)])
    
    simplex = np.zeros((n_vars + 1, n_vars))
    simplex[0] = x0
    for i in range(n_vars):
        simplex[i+1] = x0.copy()
        simplex[i+1, i] += scale * (0.5 + np.random.rand())
    
    fvals = np.array([objective(s) for s in simplex])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for iteration in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if iteration > 100 and np.max(np.abs(fvals - fvals[0])) < 1e-17:
            break
        
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)
        
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + rho * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n_vars+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n_vars+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
    
    best_x = simplex[np.argmin(fvals)]
    result = pts.copy()
    for k_idx, pi in enumerate(point_indices):
        result[pi] = [best_x[2*k_idx], best_x[2*k_idx+1]]
    return project_to_triangle(result)


def run_iterated_targeted_nm(seed, start_points=None, n_outer=100, k_triangles=3):
    """Repeatedly find bottleneck, optimize those points, repeat."""
    np.random.seed(seed)
    start_time = time.time()
    
    if start_points is None:
        start_points = BEST_R2
    
    pts = start_points.copy()
    best_metric = evaluate_config(pts)
    best_points = pts.copy()
    
    no_improve_count = 0
    
    for outer in range(n_outer):
        # Find bottleneck points
        bp = find_bottleneck_points(pts, k=k_triangles)
        scale = max(1e-4 * (0.95 ** outer), 1e-7)
        
        candidate = nelder_mead_subspace(pts, bp, scale, max_iter=30000)
        
        if check_feasibility(candidate):
            metric = evaluate_config(candidate)
            if metric > best_metric:
                best_metric = metric
                best_points = candidate.copy()
                pts = candidate.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > 15:
            # Try different k
            k_triangles = min(k_triangles + 1, 10)
            no_improve_count = 0
        
        if time.time() - start_time > 120:  # 2 min timeout per seed
            break
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'iter-tnm'}


def run_iterated_from_parent(seed):
    return run_iterated_targeted_nm(seed, start_points=PARENT_POINTS, n_outer=100, k_triangles=3)


def run_iterated_from_best(seed):
    return run_iterated_targeted_nm(seed, start_points=BEST_R2, n_outer=100, k_triangles=3)


def run_iterated_wide(seed):
    """Wider search: use more bottleneck triangles and larger scale."""
    np.random.seed(seed)
    start_time = time.time()
    
    pts = BEST_R2.copy() + np.random.randn(11, 2) * 5e-3
    pts = project_to_triangle(pts)
    best_metric = evaluate_config(BEST_R2)
    best_points = BEST_R2.copy()
    
    for outer in range(200):
        # Alternate between different k values
        k = [3, 5, 7, 10][outer % 4]
        bp = find_bottleneck_points(pts, k=k)
        scale = max(2e-3 * (0.98 ** outer), 1e-6)
        
        candidate = nelder_mead_subspace(pts, bp, scale, max_iter=20000)
        
        if check_feasibility(candidate):
            metric = evaluate_config(candidate)
            if metric > best_metric:
                best_metric = metric
                best_points = candidate.copy()
                pts = candidate.copy()
        
        if time.time() - start_time > 120:
            break
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'iter-wide'}


def run_single_config(args):
    method, seed = args
    try:
        if method == 'iter-parent':
            return run_iterated_from_parent(seed)
        elif method == 'iter-best':
            return run_iterated_from_best(seed)
        elif method == 'iter-wide':
            return run_iterated_wide(seed)
    except Exception as e:
        import traceback
        return {'seed': seed, 'metric': evaluate_config(BEST_R2), 'points': BEST_R2, 'time': 0, 'method': method, 'error': traceback.format_exc()}


if __name__ == '__main__':
    parent_metric = evaluate_config(PARENT_POINTS)
    r2_metric = evaluate_config(BEST_R2)
    print(f"Parent metric:  {parent_metric:.10f}")
    print(f"Round 2 best:   {r2_metric:.10f}")
    print(f"SOTA:           0.03653")
    
    # Analyze bottleneck of best R2
    areas = compute_all_areas(BEST_R2)
    sorted_idx = np.argsort(areas)
    print(f"\nSmallest 10 triangle areas of R2 best (normalized):")
    for i in range(10):
        idx = sorted_idx[i]
        tri = TRIPLETS[idx]
        print(f"  Triangle {tri}: area={areas[idx]/TRI_AREA:.10f}")
    print()
    
    configs = [
        ('iter-parent', 42), ('iter-parent', 123), ('iter-parent', 7),
        ('iter-best', 42), ('iter-best', 123), ('iter-best', 7),
        ('iter-wide', 42), ('iter-wide', 123), ('iter-wide', 7),
        ('iter-wide', 999), ('iter-wide', 2024), ('iter-wide', 314),
    ]
    
    print(f"Running {len(configs)} optimization configs in parallel...")
    with Pool(min(len(configs), 8)) as pool:
        results = pool.map(run_single_config, configs)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print("\n=== Results (sorted by metric) ===")
    for r in results:
        err = " ERROR" if 'error' in r else ""
        improvement = (r['metric'] - parent_metric) / parent_metric * 100
        print(f"  {r['method']:15s} seed={r['seed']:5d}  metric={r['metric']:.10f}  delta={improvement:+.4f}%  time={r['time']:.1f}s{err}")
    
    best = results[0]
    print(f"\n=== Best result ===")
    print(f"  Method: {best['method']}, Seed: {best['seed']}")
    print(f"  Metric: {best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / 0.036529889880030156:.6f}")
    print(f"  vs parent: {(best['metric'] - parent_metric) / parent_metric * 100:+.4f}%")
    print(f"  vs SOTA:   {(best['metric'] - 0.03653) / 0.03653 * 100:+.4f}%")
    print(f"\nPoints:")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'best_result.json'), 'w') as f:
        json.dump({
            'metric': best['metric'],
            'method': best['method'],
            'seed': best['seed'],
            'points': best['points'].tolist(),
            'all_results': [{'method': r['method'], 'seed': r['seed'], 'metric': r['metric'], 'time': r['time']} for r in results],
        }, f, indent=2)
