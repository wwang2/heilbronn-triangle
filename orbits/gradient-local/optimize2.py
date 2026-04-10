"""
More aggressive optimization: larger perturbations, targeted point moves,
and repeated Nelder-Mead with various scales.

Key insight from round 1: the parent is near a local optimum. The min-area
landscape has many flat directions (where the bottleneck triangle doesn't change).
We need to either:
(a) find a different basin entirely (large perturbations)
(b) carefully move points in the bottleneck triangle to equalize areas
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


def find_bottleneck_triangles(points, k=5):
    """Find the k triangles with smallest area."""
    areas = compute_all_areas(points)
    idx = np.argsort(areas)[:k]
    return idx, areas[idx]


def find_bottleneck_points(points, k=5):
    """Find which point indices participate in the smallest triangles."""
    idx, _ = find_bottleneck_triangles(points, k)
    point_set = set()
    for i in idx:
        for j in TRIPLETS[i]:
            point_set.add(j)
    return list(point_set)


# ============================================================
# Method: Targeted Nelder-Mead on bottleneck points only
# ============================================================
def run_targeted_nelder_mead(seed, n_rounds=20):
    """Move only the points involved in the smallest triangles."""
    np.random.seed(seed)
    start_time = time.time()
    
    pts = PARENT_POINTS.copy()
    best_metric = evaluate_config(pts)
    best_points = pts.copy()
    
    for round_idx in range(n_rounds):
        # Find bottleneck points
        bp = find_bottleneck_points(pts, k=3)
        n_vars = len(bp) * 2
        
        scale = 1e-3 * (0.9 ** round_idx)
        
        def objective(x_sub):
            trial = pts.copy()
            for k_idx, pi in enumerate(bp):
                trial[pi] = [x_sub[2*k_idx], x_sub[2*k_idx+1]]
            trial = project_to_triangle(trial)
            return -evaluate_config(trial)
        
        # Extract current values for bottleneck points
        x0 = np.array([pts[pi, j] for pi in bp for j in range(2)])
        
        # Run Nelder-Mead
        simplex = np.zeros((n_vars + 1, n_vars))
        simplex[0] = x0
        for i in range(n_vars):
            simplex[i+1] = x0.copy()
            simplex[i+1, i] += scale * (1 + np.random.rand())
        
        fvals = np.array([objective(s) for s in simplex])
        
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        
        for iteration in range(50000):
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            if iteration > 100 and np.max(np.abs(fvals - fvals[0])) < 1e-16:
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
        
        # Update full config
        best_x = simplex[np.argmin(fvals)]
        trial = pts.copy()
        for k_idx, pi in enumerate(bp):
            trial[pi] = [best_x[2*k_idx], best_x[2*k_idx+1]]
        trial = project_to_triangle(trial)
        
        if check_feasibility(trial):
            metric = evaluate_config(trial)
            if metric > best_metric:
                best_metric = metric
                best_points = trial.copy()
                pts = trial.copy()
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'targeted-nm'}


# ============================================================
# Method: Full Nelder-Mead with larger perturbations
# ============================================================
def run_full_nelder_mead(seed, perturbation_scale=1e-3):
    np.random.seed(seed)
    start_time = time.time()
    
    best_points = PARENT_POINTS.copy()
    best_metric = evaluate_config(best_points)
    n = 22
    
    def objective(x):
        pts = project_to_triangle(x.reshape(11, 2))
        return -evaluate_config(pts)
    
    x0 = PARENT_POINTS.flatten() + np.random.randn(n) * perturbation_scale
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += perturbation_scale * (0.5 + np.random.rand())
    
    fvals = np.array([objective(s) for s in simplex])
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    
    for iteration in range(300000):
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if iteration > 1000 and np.max(np.abs(fvals - fvals[0])) < 1e-17:
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
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
        
        # Periodic check
        if iteration % 10000 == 0:
            bx = simplex[np.argmin(fvals)]
            candidate = project_to_triangle(bx.reshape(11, 2))
            if check_feasibility(candidate):
                metric = evaluate_config(candidate)
                if metric > best_metric:
                    best_metric = metric
                    best_points = candidate.copy()
    
    bx = simplex[np.argmin(fvals)]
    candidate = project_to_triangle(bx.reshape(11, 2))
    if check_feasibility(candidate):
        metric = evaluate_config(candidate)
        if metric > best_metric:
            best_metric = metric
            best_points = candidate.copy()
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'full-nm'}


# ============================================================
# Method: Area-equalization local search
# ============================================================
def run_area_equalization(seed):
    """Try to equalize the smallest triangle areas by nudging bottleneck points."""
    np.random.seed(seed)
    start_time = time.time()
    
    pts = PARENT_POINTS.copy()
    best_metric = evaluate_config(pts)
    best_points = pts.copy()
    
    for iteration in range(100000):
        areas = compute_all_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Points in the bottleneck triangle
        tri_pts = TRIPLETS[min_idx]
        
        # Try moving each point in the bottleneck triangle
        step = 1e-4 * (0.9999 ** iteration)
        
        for pi in tri_pts:
            best_local = -1
            best_dir = None
            
            # Try random directions
            for _ in range(8):
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction)
                
                trial = pts.copy()
                trial[pi] += step * direction
                trial = project_to_triangle(trial)
                
                if check_feasibility(trial):
                    metric = evaluate_config(trial)
                    if metric > best_local:
                        best_local = metric
                        best_dir = direction
            
            if best_dir is not None and best_local > evaluate_config(pts):
                pts[pi] += step * best_dir
                pts = project_to_triangle(pts)
                
                if evaluate_config(pts) > best_metric:
                    best_metric = evaluate_config(pts)
                    best_points = pts.copy()
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'area-eq'}


def run_single_config(args):
    method, seed = args
    try:
        if method == 'targeted-nm':
            return run_targeted_nelder_mead(seed)
        elif method == 'full-nm':
            return run_full_nelder_mead(seed)
        elif method == 'area-eq':
            return run_area_equalization(seed)
    except Exception as e:
        import traceback
        return {'seed': seed, 'metric': evaluate_config(PARENT_POINTS), 'points': PARENT_POINTS, 'time': 0, 'method': method, 'error': traceback.format_exc()}


if __name__ == '__main__':
    parent_metric = evaluate_config(PARENT_POINTS)
    print(f"Parent metric: {parent_metric:.10f}")
    
    # Analyze bottleneck
    areas = compute_all_areas(PARENT_POINTS)
    sorted_areas = np.sort(areas)
    print(f"\nSmallest 5 triangle areas (normalized):")
    for i in range(5):
        idx = np.argsort(areas)[i]
        tri = TRIPLETS[idx]
        print(f"  Triangle {tri}: area={areas[idx]/TRI_AREA:.10f}")
    
    bp = find_bottleneck_points(PARENT_POINTS, k=3)
    print(f"Bottleneck points (in smallest 3 triangles): {bp}")
    print()
    
    configs = [
        ('targeted-nm', 42), ('targeted-nm', 123), ('targeted-nm', 7),
        ('targeted-nm', 999), ('targeted-nm', 2024),
        ('full-nm', 42), ('full-nm', 123), ('full-nm', 7),
        ('area-eq', 42), ('area-eq', 123), ('area-eq', 7),
    ]
    
    print(f"Running {len(configs)} optimization configs in parallel...")
    with Pool(min(len(configs), 8)) as pool:
        results = pool.map(run_single_config, configs)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print("\n=== Results (sorted by metric) ===")
    for r in results:
        err = f"  ERROR" if 'error' in r else ""
        improvement = (r['metric'] - parent_metric) / parent_metric * 100
        print(f"  {r['method']:15s} seed={r['seed']:5d}  metric={r['metric']:.10f}  delta={improvement:+.4f}%  time={r['time']:.1f}s{err}")
    
    best = results[0]
    print(f"\n=== Best result ===")
    print(f"  Method: {best['method']}, Seed: {best['seed']}")
    print(f"  Metric: {best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / 0.036529889880030156:.6f}")
    print(f"  Improvement over parent: {(best['metric'] - parent_metric) / parent_metric * 100:+.4f}%")
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
