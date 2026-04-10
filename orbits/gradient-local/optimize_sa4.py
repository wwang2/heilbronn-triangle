"""
SA4: Final push. Very long SA runs with extremely tight temperature schedule.
Also try: restart from slightly different basins found in SA2.
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

POINT_TRIPLETS = [[] for _ in range(11)]
for t_idx, (i, j, k) in enumerate(TRIPLETS):
    POINT_TRIPLETS[i].append(t_idx)
    POINT_TRIPLETS[j].append(t_idx)
    POINT_TRIPLETS[k].append(t_idx)
POINT_TRIPLETS = [np.array(pt) for pt in POINT_TRIPLETS]

BEST_SA3 = np.array([
    [0.8559678556745838, 0.0000002396801935],
    [0.6478252948064679, 0.6099843931268593],
    [0.2956488378993500, 0.0000001143876839],
    [0.4328388298412805, 0.3274481215032055],
    [0.5851015111324630, 0.1348492763781040],
    [0.5084381874446738, 0.7384883129452630],
    [0.9279835473969013, 0.1247360903895506],
    [0.3612731995118425, 0.6257435262063643],
    [0.1146692443781280, 0.0564602890110986],
    [0.6757696388215428, 0.2918845929420058],
    [0.1387253210578125, 0.2402791660411576],
])

def compute_all_areas(points):
    p = points[TRIPLETS]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )

def compute_partial_areas(points, point_idx, all_areas):
    new_areas = all_areas.copy()
    for t_idx in POINT_TRIPLETS[point_idx]:
        i, j, k = TRIPLETS[t_idx]
        a, b, c = points[i], points[j], points[k]
        new_areas[t_idx] = 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    return new_areas

def is_inside_triangle(x, y):
    return y >= 0 and y <= SQRT3 * x and y <= SQRT3 * (1 - x)

def evaluate_config(points):
    return np.min(compute_all_areas(points)) / TRI_AREA

def run_sa(seed, start_points, time_limit=300):
    np.random.seed(seed)
    start_time = time.time()
    pts = start_points.copy()
    areas = compute_all_areas(pts)
    min_area = np.min(areas)
    best_min_area = min_area
    best_points = pts.copy()

    # Multi-phase SA: warm phase then cold phase
    phases = [
        # (T0, T_final, n_steps, step_size, target_frac)
        (1e-5, 1e-7, 3000000, 5e-4, 0.6),   # warm: explore
        (1e-7, 1e-9, 3000000, 2e-4, 0.5),    # cool: refine
        (1e-9, 1e-11, 4000000, 5e-5, 0.4),   # cold: polish
    ]

    improve_count = 0
    for T0, T_final, n_steps, step_size, target_frac in phases:
        cool_rate = (T_final / T0) ** (1.0 / n_steps)
        T = T0
        accept_count = 0

        # Restart from best at each phase
        pts = best_points.copy()
        areas = compute_all_areas(pts)
        min_area = np.min(areas)

        for step in range(n_steps):
            pi = np.random.randint(11)
            if np.random.rand() < target_frac:
                min_idx = np.argmin(areas)
                tri_pts = TRIPLETS[min_idx]
                pi = tri_pts[np.random.randint(3)]

            old_pos = pts[pi].copy()
            delta = np.random.randn(2) * step_size
            new_pos = old_pos + delta

            if not is_inside_triangle(new_pos[0], new_pos[1]):
                T *= cool_rate; continue

            pts[pi] = new_pos
            new_areas = compute_partial_areas(pts, pi, areas)
            new_min = np.min(new_areas)

            delta_e = new_min - min_area
            if delta_e > 0 or np.random.rand() < np.exp(delta_e / T):
                areas = new_areas; min_area = new_min; accept_count += 1
                if min_area > best_min_area:
                    best_min_area = min_area; best_points = pts.copy(); improve_count += 1
            else:
                pts[pi] = old_pos

            T *= cool_rate
            if step % 20000 == 0 and step > 0:
                ar = accept_count / 20000
                if ar > 0.35: step_size *= 1.02
                elif ar < 0.15: step_size *= 0.98
                accept_count = 0

            if time.time() - start_time > time_limit:
                break
        if time.time() - start_time > time_limit:
            break

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_min_area / TRI_AREA, 'points': best_points, 'time': elapsed, 'improves': improve_count}

def run_config(seed):
    try:
        np.random.seed(seed)
        start = BEST_SA3.copy() + np.random.randn(11, 2) * 2e-5
        for i in range(11):
            x, y = start[i]; y = max(y, 0.0)
            if y > SQRT3*x: xp=(x+y/SQRT3)/2; y=SQRT3*xp; x=xp
            if y > SQRT3*(1-x): xp=(x-y/SQRT3+1)/2; y=SQRT3*(1-xp); x=xp
            y = max(y, 0.0); start[i] = [x, y]
        return run_sa(seed * 37 + 13, start, time_limit=300)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_SA3), 'points': BEST_SA3, 'time': 0, 'error': str(e)}

if __name__ == '__main__':
    sa3_metric = evaluate_config(BEST_SA3)
    SOTA = 0.036529889880030156
    print(f"SA3 best: {sa3_metric:.10f}")
    print(f"SOTA:     {SOTA:.10f}")
    print(f"Gap:      {sa3_metric - SOTA:.2e}")
    print()

    seeds = list(range(1, 13))
    print(f"Running {len(seeds)} multi-phase SA configs...")
    with Pool(8) as pool:
        results = pool.map(run_config, seeds)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print("\n=== Results ===")
    for r in results:
        gap = r['metric'] - SOTA
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  gap={gap:+.2e}  improves={r.get('improves',0)}  time={r['time']:.0f}s")
    
    best = results[0]
    print(f"\n=== BEST ===")
    print(f"  metric={best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / SOTA:.8f}")
    gap = best['metric'] - SOTA
    print(f"  vs SOTA: {gap:+.2e} ({gap/SOTA*100:+.6f}%)")
    if gap > 0:
        print(f"  *** BEATS SOTA! ***")
    print(f"\nPoints:")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'best_result_sa4.json'), 'w') as f:
        json.dump({'metric': best['metric'], 'seed': best['seed'], 'points': best['points'].tolist()}, f, indent=2)
