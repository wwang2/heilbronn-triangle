"""
Round 5: Refine R4 best (0.0363427) and try even wider basin search.
Two strategies:
1. Fine-tune R4 best with iterated targeted NM (small perturbations)
2. Wide search: perturbed restarts from R4 best with 0.01-0.05 scale
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

BEST_R4 = np.array([
    [0.8564134314950529, 0.0000047011056835],
    [0.6461339071990770, 0.6129140518070818],
    [0.2940623269978383, 0.0000000000000000],
    [0.4314701345699793, 0.3286060481382833],
    [0.5802958352289830, 0.1362415980979603],
    [0.5078300233868074, 0.7402711262921544],
    [0.9265099084020968, 0.1268228569731426],
    [0.3630403939174118, 0.6288044074647765],
    [0.1163059267196851, 0.0578206246295111],
    [0.6749535555484554, 0.2950670696246775],
    [0.1396768159335244, 0.2394717817882160],
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
    
    for iteration in range(max_iter):
        order = np.argsort(fvals)
        simplex = simplex[order]; fvals = fvals[order]
        if iteration > 100 and np.max(np.abs(fvals - fvals[0])) < 1e-17:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1]); fr = objective(xr)
        if fr < fvals[0]:
            xe = centroid + 2*(xr - centroid); fe = objective(xe)
            if fe < fr: simplex[-1], fvals[-1] = xe, fe
            else: simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            if fr < fvals[-1]:
                xc = centroid + 0.5*(xr - centroid); fc = objective(xc)
                if fc <= fr: simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n_vars+1):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                xc = centroid - 0.5*(centroid - simplex[-1]); fc = objective(xc)
                if fc < fvals[-1]: simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, n_vars+1):
                        simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
    
    best_x = simplex[np.argmin(fvals)]
    result = pts.copy()
    for k_idx, pi in enumerate(point_indices):
        result[pi] = [best_x[2*k_idx], best_x[2*k_idx+1]]
    return project_to_triangle(result)


def run_deep_iterated(seed, start_points, time_limit=240):
    np.random.seed(seed)
    start_time = time.time()
    pts = start_points.copy()
    best_metric = evaluate_config(pts)
    best_points = pts.copy()
    stale = 0; k = 3; iteration = 0
    
    while time.time() - start_time < time_limit:
        bp = find_bottleneck_points(pts, k=k)
        scale = max(1e-3 * (0.97 ** iteration), 1e-7)
        candidate = nelder_mead_subspace(pts, bp, scale, max_iter=30000)
        
        if check_feasibility(candidate):
            metric = evaluate_config(candidate)
            if metric > best_metric + 1e-12:
                best_metric = metric; best_points = candidate.copy(); pts = candidate.copy(); stale = 0
            else: stale += 1
        else: stale += 1
        
        if stale > 5:
            k = min(k + 1, 15); stale = 0
        if stale > 10:
            pts = best_points.copy() + np.random.randn(11, 2) * 5e-4
            pts = project_to_triangle(pts); k = 3; stale = 0
        iteration += 1
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'deep-iter'}


def run_config(args):
    method, seed, perturb_scale = args
    try:
        np.random.seed(seed)
        if method == 'refine-r4':
            start = BEST_R4.copy()
            return run_deep_iterated(seed, start, time_limit=240)
        elif method == 'perturb-r4':
            start = BEST_R4.copy() + np.random.randn(11, 2) * perturb_scale
            start = project_to_triangle(start)
            return run_deep_iterated(seed * 13 + 5, start, time_limit=240)
        elif method == 'perturb-parent':
            start = PARENT_POINTS.copy() + np.random.randn(11, 2) * perturb_scale
            start = project_to_triangle(start)
            return run_deep_iterated(seed * 17 + 11, start, time_limit=240)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_R4), 'points': BEST_R4, 'time': 0, 'method': method, 'error': str(e)}


if __name__ == '__main__':
    r4_metric = evaluate_config(BEST_R4)
    print(f"R4 best metric: {r4_metric:.10f}")
    print(f"SOTA:           0.03653")
    
    areas = compute_all_areas(BEST_R4)
    sorted_idx = np.argsort(areas)
    print(f"\nSmallest 5 triangle areas of R4 best (normalized):")
    for i in range(5):
        idx = sorted_idx[i]
        tri = TRIPLETS[idx]
        print(f"  Triangle {tri}: area={areas[idx]/TRI_AREA:.10f}")
    print()
    
    configs = [
        # Refine R4
        ('refine-r4', 42, 0), ('refine-r4', 123, 0), ('refine-r4', 7, 0),
        # Perturb R4 small
        ('perturb-r4', 42, 5e-3), ('perturb-r4', 123, 5e-3), ('perturb-r4', 7, 5e-3),
        # Perturb R4 medium
        ('perturb-r4', 999, 1e-2), ('perturb-r4', 2024, 1e-2), ('perturb-r4', 314, 1e-2),
        # Perturb R4 large
        ('perturb-r4', 555, 3e-2), ('perturb-r4', 777, 3e-2),
        # Perturb parent large
        ('perturb-parent', 42, 1e-2), ('perturb-parent', 123, 1e-2),
    ]
    
    print(f"Running {len(configs)} configs in parallel...")
    with Pool(min(len(configs), 8)) as pool:
        results = pool.map(run_config, configs)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    parent_metric = evaluate_config(PARENT_POINTS)
    print("\n=== Results (sorted by metric) ===")
    for r in results:
        imp = (r['metric'] - parent_metric) / parent_metric * 100
        gap = (r['metric'] - 0.03653) / 0.03653 * 100
        print(f"  {r['method']:18s} seed={r['seed']:5d}  metric={r['metric']:.10f}  vs_parent={imp:+.4f}%  vs_SOTA={gap:+.4f}%  time={r['time']:.0f}s")
    
    best = results[0]
    print(f"\n=== BEST ===")
    print(f"  {best['method']} seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / 0.036529889880030156:.6f}")
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
