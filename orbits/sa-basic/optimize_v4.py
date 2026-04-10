"""
SA optimization v4: numba-JIT compiled inner loop.
Reduced parameters to fit in time budget.
"""
import sys
import numpy as np
import numba as nb
import time
import os
import itertools

EQUILATERAL_AREA = 0.5 * np.sqrt(3) / 2
N_POINTS = 11

ALL_TRIPLETS = np.array(list(itertools.combinations(range(N_POINTS), 3)), dtype=np.int32)

_pt = []
for i in range(N_POINTS):
    mask = np.any(ALL_TRIPLETS == i, axis=1)
    _pt.append(np.where(mask)[0].astype(np.int32))
MAX_TRI_PP = max(len(x) for x in _pt)
PT_TRIPLETS = np.full((N_POINTS, MAX_TRI_PP), -1, dtype=np.int32)
PT_COUNT = np.zeros(N_POINTS, dtype=np.int32)
for i in range(N_POINTS):
    PT_COUNT[i] = len(_pt[i])
    PT_TRIPLETS[i, :len(_pt[i])] = _pt[i]


@nb.njit(cache=True)
def project_to_triangle(x, y):
    sqrt3 = 1.7320508075688772
    if y < 0.0: y = 0.0
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        if t < 0.0: t = 0.0
        elif t > 0.5: t = 0.5
        x, y = t, sqrt3 * t
    if y > sqrt3 * (1.0 - x):
        t = (x - y / sqrt3 + 1.0) / 2.0
        if t < 0.5: t = 0.5
        elif t > 1.0: t = 1.0
        x, y = t, sqrt3 * (1.0 - t)
    if y < 0.0: y = 0.0
    return x, y


@nb.njit(cache=True)
def compute_all_areas(points, triplets, areas):
    for k in range(triplets.shape[0]):
        i0, i1, i2 = triplets[k, 0], triplets[k, 1], triplets[k, 2]
        areas[k] = 0.5 * abs(
            points[i0, 0] * (points[i1, 1] - points[i2, 1]) +
            points[i1, 0] * (points[i2, 1] - points[i0, 1]) +
            points[i2, 0] * (points[i0, 1] - points[i1, 1]))


@nb.njit(cache=True)
def update_areas(points, triplets, areas, pt_tri, pt_cnt, idx):
    for j in range(pt_cnt[idx]):
        k = pt_tri[idx, j]
        i0, i1, i2 = triplets[k, 0], triplets[k, 1], triplets[k, 2]
        areas[k] = 0.5 * abs(
            points[i0, 0] * (points[i1, 1] - points[i2, 1]) +
            points[i1, 0] * (points[i2, 1] - points[i0, 1]) +
            points[i2, 0] * (points[i0, 1] - points[i1, 1]))


@nb.njit(cache=True)
def sa_core(seed, max_iter, n_restarts, triplets, pt_tri, pt_cnt):
    n_tri = triplets.shape[0]
    max_pp = pt_tri.shape[1]
    sqrt3h = 0.8660254037844386
    
    overall_best = -1.0
    overall_best_pts = np.zeros((11, 2))
    
    for restart in range(n_restarts):
        np.random.seed(seed + restart * 997)
        pts = np.zeros((11, 2))
        for i in range(11):
            r1, r2 = np.random.random(), np.random.random()
            if r1 + r2 > 1.0: r1, r2 = 1.0 - r1, 1.0 - r2
            pts[i, 0] = r1 + r2 * 0.5
            pts[i, 1] = r2 * sqrt3h
        
        areas = np.zeros(n_tri)
        compute_all_areas(pts, triplets, areas)
        cur_min = areas.min()
        best = cur_min
        best_pts = pts.copy()
        
        T0, Tf = 0.015, 1e-9
        cool = (Tf / T0) ** (1.0 / max_iter)
        ps0, psf = 0.08, 0.0005
        pr = (psf / ps0) ** (1.0 / max_iter)
        T, ps = T0, ps0
        no_imp = 0
        buf = np.zeros(max_pp)
        
        for it in range(max_iter):
            idx = np.random.randint(0, 11)
            ox, oy = pts[idx, 0], pts[idx, 1]
            cnt = pt_cnt[idx]
            for j in range(cnt): buf[j] = areas[pt_tri[idx, j]]
            
            nx, ny = project_to_triangle(ox + np.random.randn() * ps, oy + np.random.randn() * ps)
            pts[idx, 0], pts[idx, 1] = nx, ny
            update_areas(pts, triplets, areas, pt_tri, pt_cnt, idx)
            new_min = areas.min()
            delta = new_min - cur_min
            
            if delta > 0:
                cur_min = new_min
                no_imp = 0
                if new_min > best:
                    best = new_min
                    best_pts = pts.copy()
            elif T > 0 and np.random.random() < np.exp(delta / T):
                cur_min = new_min
                no_imp += 1
            else:
                pts[idx, 0], pts[idx, 1] = ox, oy
                for j in range(cnt): areas[pt_tri[idx, j]] = buf[j]
                no_imp += 1
            
            T *= cool
            ps *= pr
            if no_imp > 5000:
                T = max(T, T0 * 0.03)
                ps = max(ps, ps0 * 0.15)
                no_imp = 0
        
        # Refinement
        pts[:] = best_pts
        compute_all_areas(pts, triplets, areas)
        cur_min = best
        for ri in range(100000):
            idx = np.random.randint(0, 11)
            ox, oy = pts[idx, 0], pts[idx, 1]
            cnt = pt_cnt[idx]
            for j in range(cnt): buf[j] = areas[pt_tri[idx, j]]
            sc = 0.001 * (1.0 - ri / 100000.0) + 0.00005
            nx, ny = project_to_triangle(ox + np.random.randn() * sc, oy + np.random.randn() * sc)
            pts[idx, 0], pts[idx, 1] = nx, ny
            update_areas(pts, triplets, areas, pt_tri, pt_cnt, idx)
            new_min = areas.min()
            if new_min > cur_min:
                cur_min = new_min
                if new_min > best:
                    best = new_min
                    best_pts = pts.copy()
            else:
                pts[idx, 0], pts[idx, 1] = ox, oy
                for j in range(cnt): areas[pt_tri[idx, j]] = buf[j]
        
        if best > overall_best:
            overall_best = best
            overall_best_pts = best_pts.copy()
    
    return overall_best, overall_best_pts


def format_points_array(points):
    lines = ["np.array(["]
    for i, (x, y) in enumerate(points):
        comma = "," if i < len(points) - 1 else ""
        lines.append(f"    [{x:.15f}, {y:.15f}]{comma}")
    lines.append("])")
    return "\n".join(lines)


def update_solution_file(points, metric):
    import re
    solution_path = os.path.join(os.path.dirname(__file__), "solution.py")
    with open(solution_path, 'r') as f:
        content = f.read()
    pts_str = format_points_array(points)
    new_assign = f"BEST_POINTS = {pts_str}"
    if "BEST_POINTS = None" in content:
        content = content.replace("BEST_POINTS = None", new_assign)
    else:
        pattern = r'BEST_POINTS = np\.array\(\[.*?\]\)'
        content = re.sub(pattern, new_assign, content, flags=re.DOTALL)
    with open(solution_path, 'w') as f:
        f.write(content)
    print(f"Updated solution.py (metric={metric:.6f})", flush=True)


if __name__ == '__main__':
    print("Warming up JIT...", flush=True)
    _ = sa_core(0, 100, 1, ALL_TRIPLETS, PT_TRIPLETS, PT_COUNT)
    print("JIT ready.", flush=True)
    
    seeds = [42, 123, 7]
    print(f"Running SA v4 (numba): {len(seeds)} seeds, 20 restarts x 1M iters + 100k refine each", flush=True)
    
    t_total = time.time()
    results = []
    for s in seeds:
        t0 = time.time()
        obj, pts = sa_core(s, 1000000, 20, ALL_TRIPLETS, PT_TRIPLETS, PT_COUNT)
        elapsed = time.time() - t0
        norm = obj / EQUILATERAL_AREA
        print(f"  Seed {s}: metric={norm:.6f} time={elapsed:.1f}s", flush=True)
        results.append({'seed': s, 'min_area_normalized': norm, 'points': pts, 'time': elapsed})
    t_total = time.time() - t_total
    
    print(f"\n{'Seed':>8} | {'Metric':>10} | {'Time (s)':>8}", flush=True)
    print("-" * 35, flush=True)
    metrics = [r['min_area_normalized'] for r in results]
    for r in results:
        print(f"{r['seed']:>8} | {r['min_area_normalized']:>10.6f} | {r['time']:>8.1f}", flush=True)
    mean_m, std_m = np.mean(metrics), np.std(metrics)
    print("-" * 35, flush=True)
    print(f"{'Mean':>8} | {mean_m:>10.6f} +/- {std_m:.6f}", flush=True)
    print(f"Total: {t_total:.1f}s", flush=True)
    
    best = max(results, key=lambda r: r['min_area_normalized'])
    print(f"\nBest: seed={best['seed']}, metric={best['min_area_normalized']:.6f}, score={best['min_area_normalized']/0.036530:.4f}", flush=True)
    update_solution_file(best['points'], best['min_area_normalized'])
