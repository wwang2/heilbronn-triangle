"""
Simulated annealing from R6 best to try to escape basin.
SA can accept worse solutions probabilistically, which NM cannot.
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

# Precompute which triplets each point participates in
POINT_TRIPLETS = [[] for _ in range(11)]
for t_idx, (i, j, k) in enumerate(TRIPLETS):
    POINT_TRIPLETS[i].append(t_idx)
    POINT_TRIPLETS[j].append(t_idx)
    POINT_TRIPLETS[k].append(t_idx)
POINT_TRIPLETS = [np.array(pt) for pt in POINT_TRIPLETS]

BEST_R6 = np.array([
    [0.8565490556373909, 0.0000000000000000],
    [0.6467315294564712, 0.6118789389040159],
    [0.2961788657273172, 0.0000000000000000],
    [0.4331309614910825, 0.3290419304422803],
    [0.5843510648541337, 0.1363494260745345],
    [0.5080244116741026, 0.7393423878620344],
    [0.9266632016691652, 0.1270230607734384],
    [0.3624338457504128, 0.6277538352222964],
    [0.1160975777143270, 0.0569436357775985],
    [0.6751289977675210, 0.2943353312528431],
    [0.1400950584756128, 0.2406484948435915],
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
    """Recompute only areas involving point_idx. Returns updated areas array."""
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


def run_sa(seed, time_limit=240):
    np.random.seed(seed)
    start_time = time.time()

    pts = BEST_R6.copy()
    areas = compute_all_areas(pts)
    min_area = np.min(areas)
    
    best_min_area = min_area
    best_points = pts.copy()
    
    # SA parameters
    T0 = 1e-5  # Initial temperature (scale of area values)
    T_final = 1e-9
    n_steps = 5000000
    
    # Cooling: geometric
    cool_rate = (T_final / T0) ** (1.0 / n_steps)
    T = T0
    
    step_size = 1e-3  # Initial step size
    accept_count = 0
    improve_count = 0
    
    for step in range(n_steps):
        # Choose random point
        pi = np.random.randint(11)
        
        # Targeted: with 60% probability, choose from bottleneck triangle
        if np.random.rand() < 0.6:
            min_idx = np.argmin(areas)
            tri_pts = TRIPLETS[min_idx]
            pi = tri_pts[np.random.randint(3)]
        
        # Propose move
        old_pos = pts[pi].copy()
        delta = np.random.randn(2) * step_size
        new_pos = old_pos + delta
        
        if not is_inside_triangle(new_pos[0], new_pos[1]):
            T *= cool_rate
            continue
        
        # Compute new areas (partial update)
        pts[pi] = new_pos
        new_areas = compute_partial_areas(pts, pi, areas)
        new_min = np.min(new_areas)
        
        # Acceptance criterion
        delta_e = new_min - min_area  # positive = improvement
        
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
        
        # Adaptive step size every 10000 steps
        if step % 10000 == 0 and step > 0:
            accept_rate = accept_count / 10000
            if accept_rate > 0.4:
                step_size *= 1.1
            elif accept_rate < 0.2:
                step_size *= 0.9
            accept_count = 0
        
        if time.time() - start_time > time_limit:
            break
    
    elapsed = time.time() - start_time
    return {
        'seed': seed,
        'metric': best_min_area / TRI_AREA,
        'points': best_points,
        'time': elapsed,
        'method': 'sa',
        'improves': improve_count,
    }


def run_config(seed):
    try:
        return run_sa(seed, time_limit=240)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(BEST_R6), 'points': BEST_R6, 'time': 0, 'method': 'sa', 'error': str(e)}


if __name__ == '__main__':
    r6_metric = evaluate_config(BEST_R6)
    print(f"R6 best metric: {r6_metric:.10f}")
    print(f"SOTA:           0.03653")
    print()
    
    seeds = list(range(1, 13))
    
    print(f"Running {len(seeds)} SA configs in parallel...")
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
    with open(os.path.join(outdir, 'best_result_sa.json'), 'w') as f:
        json.dump({
            'metric': best['metric'],
            'seed': best['seed'],
            'points': best['points'].tolist(),
        }, f, indent=2)
