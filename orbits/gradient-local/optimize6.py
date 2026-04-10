"""
Round 6: Refine R5 best (0.036389) and do massive restart search.
The trend is clear: perturbed restarts find better basins.
Use more seeds and a range of perturbation scales.
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

BEST_R5 = np.array([
    [0.8565352797521506, 0.0000000000000000],
    [0.6465171167879542, 0.6122503133291990],
    [0.2958218192088939, 0.0000000000000000],
    [0.4327344586586153, 0.3288869376737000],
    [0.5836173619789998, 0.1362628801688295],
    [0.5076683482389313, 0.7390681554322120],
    [0.9268238346135829, 0.1267434244604947],
    [0.3623499173767359, 0.6276084670148914],
    [0.1166797043074680, 0.0573796216754756],
    [0.6749160839385182, 0.2944238576182466],
    [0.1402367036583184, 0.2403837646009557],
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


def run_deep_iterated(seed, start_points, time_limit=200):
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
            k = min(k + 2, 15); stale = 0
        if stale > 10:
            pts = best_points.copy() + np.random.randn(11, 2) * 5e-4
            pts = project_to_triangle(pts); k = 3; stale = 0
        iteration += 1
    
    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'deep-iter'}


def run_config(args):
    seed, perturb_scale = args
    try:
        np.random.seed(seed)
        start = BEST_R5.copy() + np.random.randn(11, 2) * perturb_scale
        start = project_to_triangle(start)
        result = run_deep_iterated(seed * 13 + 7, start, time_limit=200)
        result['perturb_scale'] = perturb_scale
        return result
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_R5), 'points': BEST_R5, 'time': 0, 'method': 'error', 'error': str(e)}


if __name__ == '__main__':
    r5_metric = evaluate_config(BEST_R5)
    print(f"R5 best metric: {r5_metric:.10f}")
    print(f"SOTA:           0.03653")
    print()
    
    # Many seeds at the productive perturbation scales
    configs = []
    # Fine refinement from R5
    for s in [42, 123, 7, 999, 2024]:
        configs.append((s, 1e-4))
    # Small perturbation 
    for s in [42, 123, 7, 999, 2024, 314, 555, 777]:
        configs.append((s, 3e-3))
    # Medium perturbation (best scale from R4/R5)
    for s in range(10):
        configs.append((s * 100 + 1, 5e-3))
    # Larger
    for s in range(5):
        configs.append((s * 1000 + 2, 1e-2))
    
    print(f"Running {len(configs)} configs in parallel...")
    with Pool(8) as pool:
        results = pool.map(run_config, configs)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print("\n=== Top 10 Results ===")
    for r in results[:10]:
        gap = (r['metric'] - 0.03653) / 0.03653 * 100
        ps = r.get('perturb_scale', 0)
        print(f"  seed={r['seed']:5d}  scale={ps:.0e}  metric={r['metric']:.10f}  vs_SOTA={gap:+.4f}%  time={r['time']:.0f}s")
    
    print(f"\n=== Worst 3 ===")
    for r in results[-3:]:
        gap = (r['metric'] - 0.03653) / 0.03653 * 100
        print(f"  seed={r['seed']:5d}  metric={r['metric']:.10f}  vs_SOTA={gap:+.4f}%")
    
    best = results[0]
    print(f"\n=== BEST ===")
    print(f"  seed={best['seed']}  metric={best['metric']:.10f}")
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
