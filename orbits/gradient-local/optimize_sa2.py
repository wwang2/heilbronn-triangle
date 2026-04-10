"""
Round SA2: Refine SA best (0.036528) with more SA iterations and smaller temperature.
Also try SA from multiple R6/SA seeds.
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

BEST_SA = np.array([
    [0.8559272981320613, 0.0000006220278139],
    [0.6478039697350225, 0.6100150956905622],
    [0.2956687576357860, 0.0000016042018787],
    [0.4328654189749260, 0.3275063375485437],
    [0.5851633714704966, 0.1348326752769101],
    [0.5084380847933188, 0.7384901973657931],
    [0.9280045620427737, 0.1246989709589986],
    [0.3612919293748665, 0.6257709908036870],
    [0.1146755961970737, 0.0564654200322581],
    [0.6757156699856179, 0.2919443422410194],
    [0.1387442487776641, 0.2403012933746593],
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
            a[0] * (b[1] - c[1]) +
            b[0] * (c[1] - a[1]) +
            c[0] * (a[1] - b[1])
        )
    return new_areas

def is_inside_triangle(x, y, tol=0.0):
    return y >= -tol and y <= SQRT3 * x + tol and y <= SQRT3 * (1 - x) + tol

def evaluate_config(points):
    return np.min(compute_all_areas(points)) / TRI_AREA


def run_sa_refined(seed, start_points, time_limit=280):
    np.random.seed(seed)
    start_time = time.time()

    pts = start_points.copy()
    areas = compute_all_areas(pts)
    min_area = np.min(areas)
    best_min_area = min_area
    best_points = pts.copy()

    # Tighter SA: start from lower T since we're already near optimum
    T0 = 5e-6
    T_final = 1e-10
    n_steps = 8000000
    cool_rate = (T_final / T0) ** (1.0 / n_steps)
    T = T0
    step_size = 5e-4
    accept_count = 0
    improve_count = 0

    for step in range(n_steps):
        pi = np.random.randint(11)
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
            areas = new_areas
            min_area = new_min
            accept_count += 1
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = pts.copy()
                improve_count += 1
        else:
            pts[pi] = old_pos

        T *= cool_rate

        if step % 20000 == 0 and step > 0:
            accept_rate = accept_count / 20000
            if accept_rate > 0.35:
                step_size *= 1.05
            elif accept_rate < 0.15:
                step_size *= 0.95
            accept_count = 0

        if time.time() - start_time > time_limit:
            break

    elapsed = time.time() - start_time
    return {
        'seed': seed, 'metric': best_min_area / TRI_AREA,
        'points': best_points, 'time': elapsed,
        'improves': improve_count,
    }


def run_config(seed):
    try:
        # Small perturbation from SA best
        np.random.seed(seed)
        start = BEST_SA.copy() + np.random.randn(11, 2) * 1e-4
        # Project to triangle
        for i in range(11):
            x, y = start[i]
            y = max(y, 0.0)
            if y > SQRT3 * x:
                xp = (x + y / SQRT3) / 2.0; y = SQRT3 * xp; x = xp
            if y > SQRT3 * (1 - x):
                xp = (x - y / SQRT3 + 1) / 2.0; y = SQRT3 * (1 - xp); x = xp
            y = max(y, 0.0)
            start[i] = [x, y]
        return run_sa_refined(seed * 23 + 7, start, time_limit=280)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_SA), 'points': BEST_SA, 'time': 0, 'error': str(e)}


if __name__ == '__main__':
    sa_metric = evaluate_config(BEST_SA)
    print(f"SA best metric: {sa_metric:.10f}")
    print(f"SOTA:           0.03653")
    print()

    seeds = list(range(1, 13))
    print(f"Running {len(seeds)} refined SA configs in parallel...")
    with Pool(8) as pool:
        results = pool.map(run_config, seeds)

    results.sort(key=lambda r: r['metric'], reverse=True)

    print("\n=== Results ===")
    for r in results:
        gap = (r['metric'] - 0.03653) / 0.03653 * 100
        imp = r.get('improves', 0)
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  vs_SOTA={gap:+.4f}%  improves={imp}  time={r['time']:.0f}s")

    best = results[0]
    print(f"\n=== BEST ===")
    print(f"  seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"  Combined score: {best['metric'] / 0.036529889880030156:.6f}")
    print(f"  vs SOTA: {(best['metric'] - 0.03653) / 0.03653 * 100:+.4f}%")
    print(f"\nPoints:")
    for p in best['points']:
        print(f"    [{p[0]:.16f}, {p[1]:.16f}],")

    outdir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(outdir, 'best_result_sa2.json'), 'w') as f:
        json.dump({
            'metric': best['metric'],
            'seed': best['seed'],
            'points': best['points'].tolist(),
        }, f, indent=2)
