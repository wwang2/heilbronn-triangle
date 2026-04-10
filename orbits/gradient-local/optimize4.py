"""
Round 4: Multi-restart from best R3. Use iterated targeted NM with varied 
perturbation sizes and both parent/R3 starts. Focus on longer runs.
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

BEST_R3 = np.array([
    [0.8560396341392511, 0.0000000000000000],
    [0.6449975475606342, 0.6148822844365355],
    [0.2902037951445892, 0.0000036222496365],
    [0.4292401917127936, 0.3296258086031366],
    [0.5759849252914653, 0.1352959921574288],
    [0.5074496157952484, 0.7412916100960815],
    [0.9262885858842926, 0.1276719143461549],
    [0.3639928191733857, 0.6304540563985350],
    [0.1116998572298332, 0.0556008495394041],
    [0.6736430789185808, 0.2976335013960326],
    [0.1355625543589845, 0.2346859830140469],
])

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


def run_deep_iterated(seed, start_points, time_limit=180):
    """Deep iterated targeted NM with adaptive k and scale."""
    np.random.seed(seed)
    start_time = time.time()
    
    pts = start_points.copy()
    best_metric = evaluate_config(pts)
    best_points = pts.copy()
    
    stale = 0
    k = 3
    scale_base = 1e-3
    
    iteration = 0
    while time.time() - start_time < time_limit:
        bp = find_bottleneck_points(pts, k=k)
        scale = scale_base * (0.98 ** iteration)
        scale = max(scale, 1e-7)
        
        candidate = nelder_mead_subspace(pts, bp, scale, max_iter=30000)
        
        if check_feasibility(candidate):
            metric = evaluate_config(candidate)
            if metric > best_metric + 1e-12:
                best_metric = metric
                best_points = candidate.copy()
                pts = candidate.copy()
                stale = 0
            else:
                stale += 1
        else:
            stale += 1
        
        # Adaptive strategy
        if stale > 5:
            k = min(k + 1, 15)
            stale = 0
        if stale > 10:
            # Random restart near best
            pts = best_points.copy() + np.random.randn(11, 2) * 1e-3
            pts = project_to_triangle(pts)
            k = 3
            stale = 0
        
        iteration += 1
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'deep-iter'}


def run_perturbed_restart(seed, time_limit=180):
    """Start from perturbed R3 best, do deep iterated NM."""
    np.random.seed(seed)
    
    # Larger perturbation to escape basin
    start = BEST_R3.copy() + np.random.randn(11, 2) * 5e-3
    start = project_to_triangle(start)
    
    return run_deep_iterated(seed * 7 + 3, start, time_limit)


def run_from_r3(seed, time_limit=180):
    return run_deep_iterated(seed, BEST_R3.copy(), time_limit)


def run_from_parent(seed, time_limit=180):
    return run_deep_iterated(seed, PARENT_POINTS.copy(), time_limit)


def run_single_config(args):
    method, seed = args
    try:
        if method == 'from-r3':
            return run_from_r3(seed)
        elif method == 'from-parent':
            return run_from_parent(seed)
        elif method == 'perturbed':
            return run_perturbed_restart(seed)
    except Exception as e:
        import traceback
        return {'seed': seed, 'metric': evaluate_config(BEST_R3), 'points': BEST_R3, 'time': 0, 'method': method, 'error': traceback.format_exc()}


if __name__ == '__main__':
    r3_metric = evaluate_config(BEST_R3)
    print(f"R3 best metric: {r3_metric:.10f}")
    print(f"SOTA:           0.03653")
    print()
    
    configs = [
        ('from-r3', 42), ('from-r3', 123), ('from-r3', 7),
        ('from-parent', 42), ('from-parent', 123), ('from-parent', 7),
        ('perturbed', 42), ('perturbed', 123), ('perturbed', 7),
        ('perturbed', 999), ('perturbed', 2024), ('perturbed', 314),
    ]
    
    print(f"Running {len(configs)} configs in parallel...")
    with Pool(min(len(configs), 8)) as pool:
        results = pool.map(run_single_config, configs)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    parent_metric = evaluate_config(PARENT_POINTS)
    print("\n=== Results (sorted by metric) ===")
    for r in results:
        err = " ERR" if 'error' in r else ""
        imp = (r['metric'] - parent_metric) / parent_metric * 100
        print(f"  {r['method']:15s} seed={r['seed']:5d}  metric={r['metric']:.10f}  delta={imp:+.4f}%  time={r['time']:.1f}s{err}")
    
    best = results[0]
    print(f"\n=== Best result ===")
    print(f"  Method: {best['method']}, Seed: {best['seed']}")
    print(f"  Metric: {best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / 0.036529889880030156:.6f}")
    print(f"  vs SOTA: {(best['metric'] - 0.03653) / 0.03653 * 100:+.4f}%")
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
