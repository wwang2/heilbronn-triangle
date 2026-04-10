"""
Round SA3: Push past SOTA. Start from SA2 best (0.036530), use very tight SA.
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

BEST_SA2 = np.array([
    [0.8559674182467510, 0.0000004079217227],
    [0.6478240613570523, 0.6099863753455416],
    [0.2956509792463695, 0.0000000049806978],
    [0.4328410232449911, 0.3274519424798367],
    [0.5851056043899155, 0.1348494772677565],
    [0.5084382481416670, 0.7384885992334003],
    [0.9279835016489855, 0.1247358122966673],
    [0.3612744616731405, 0.6257456627294864],
    [0.1146697725729046, 0.0564603635994206],
    [0.6757680571952279, 0.2918878571043867],
    [0.1387269396519449, 0.2402814139198642],
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
        new_areas[t_idx] = 0.5 * abs(
            a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
        )
    return new_areas

def is_inside_triangle(x, y):
    return y >= 0 and y <= SQRT3 * x and y <= SQRT3 * (1 - x)

def evaluate_config(points):
    return np.min(compute_all_areas(points)) / TRI_AREA

def run_sa_final(seed, start_points, time_limit=300):
    np.random.seed(seed)
    start_time = time.time()
    pts = start_points.copy()
    areas = compute_all_areas(pts)
    min_area = np.min(areas)
    best_min_area = min_area
    best_points = pts.copy()

    # Very fine SA
    T0 = 2e-6
    T_final = 1e-11
    n_steps = 10000000
    cool_rate = (T_final / T0) ** (1.0 / n_steps)
    T = T0
    step_size = 3e-4
    accept_count = 0
    improve_count = 0

    for step in range(n_steps):
        pi = np.random.randint(11)
        # Targeted: 50% from bottleneck
        if np.random.rand() < 0.5:
            min_idx = np.argmin(areas)
            tri_pts = TRIPLETS[min_idx]
            pi = tri_pts[np.random.randint(3)]

        old_pos = pts[pi].copy()
        delta = np.random.randn(2) * step_size
        new_pos = old_pos + delta

        if not is_inside_triangle(new_pos[0], new_pos[1]):
            T *= cool_rate
            continue

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
            if ar > 0.35: step_size *= 1.03
            elif ar < 0.15: step_size *= 0.97
            accept_count = 0

        if time.time() - start_time > time_limit:
            break

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_min_area / TRI_AREA, 'points': best_points, 'time': elapsed, 'improves': improve_count}

def run_config(seed):
    try:
        np.random.seed(seed)
        start = BEST_SA2.copy() + np.random.randn(11, 2) * 5e-5
        for i in range(11):
            x, y = start[i]
            y = max(y, 0.0)
            if y > SQRT3 * x: xp = (x + y/SQRT3)/2; y = SQRT3*xp; x = xp
            if y > SQRT3 * (1-x): xp = (x - y/SQRT3 + 1)/2; y = SQRT3*(1-xp); x = xp
            y = max(y, 0.0); start[i] = [x, y]
        return run_sa_final(seed * 31 + 11, start, time_limit=300)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_SA2), 'points': BEST_SA2, 'time': 0, 'error': str(e)}

if __name__ == '__main__':
    sa2_metric = evaluate_config(BEST_SA2)
    print(f"SA2 best:  {sa2_metric:.10f}")
    print(f"SOTA:      0.036529889880030156")
    print(f"Gap:       {(sa2_metric - 0.036529889880030156):.15f}")
    print()

    seeds = list(range(1, 13))
    print(f"Running {len(seeds)} final SA configs...")
    with Pool(8) as pool:
        results = pool.map(run_config, seeds)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    SOTA = 0.036529889880030156
    print("\n=== Results ===")
    for r in results:
        gap = r['metric'] - SOTA
        pct = gap / SOTA * 100
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  gap={gap:+.2e}  ({pct:+.4f}%)  improves={r.get('improves',0)}  time={r['time']:.0f}s")
    
    best = results[0]
    print(f"\n=== BEST ===")
    print(f"  metric={best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / SOTA:.8f}")
    print(f"  vs SOTA: {(best['metric'] - SOTA) / SOTA * 100:+.6f}%")
    print(f"\nPoints:")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")
    
    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'best_result_sa3.json'), 'w') as f:
        json.dump({'metric': best['metric'], 'seed': best['seed'], 'points': best['points'].tolist()}, f, indent=2)
