"""
Refine the best SA result (metric=0.0363) with focused local optimization.
Uses SA with small step sizes and many iterations, starting from the known good config.
"""
import numpy as np
from itertools import combinations
import multiprocessing as mp
import json
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2
TRI_AREA = H / 2

TRIPLETS = np.array(list(combinations(range(N), 3)))
I0, I1, I2 = TRIPLETS[:, 0], TRIPLETS[:, 1], TRIPLETS[:, 2]

POINT_TRIPLET_IDX = []
for p in range(N):
    mask = (I0 == p) | (I1 == p) | (I2 == p)
    POINT_TRIPLET_IDX.append(np.where(mask)[0])


def is_inside(x, y):
    return y >= -1e-12 and SQRT3*x >= y - 1e-12 and SQRT3*(1-x) >= y - 1e-12

def project(x, y):
    if is_inside(x, y):
        return x, y
    lam3 = y / H
    lam2 = x - 0.5 * lam3
    lam1 = 1.0 - lam3 - lam2
    lam1 = max(0, lam1); lam2 = max(0, lam2); lam3 = max(0, lam3)
    s = lam1 + lam2 + lam3
    if s < 1e-15: return 0.5, H/3
    lam1 /= s; lam2 /= s; lam3 /= s
    return lam2 + 0.5*lam3, H*lam3

def compute_all_areas(points):
    a, b, c = points[I0], points[I1], points[I2]
    return 0.5 * np.abs(
        a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1])
    )

def update_areas(points, idx, areas):
    new_areas = areas.copy()
    for ti in POINT_TRIPLET_IDX[idx]:
        i, j, k = TRIPLETS[ti]
        a, b, c = points[i], points[j], points[k]
        new_areas[ti] = 0.5 * abs(a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    return new_areas

# Best configuration from SA (metric=0.0363)
BEST_POINTS = np.array([
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


def refine_sa(seed, total_iters=10_000_000):
    rng = np.random.RandomState(seed)
    
    # Start from best known + small perturbation
    points = BEST_POINTS.copy()
    perturb = 0.005 * seed  # different perturbation levels
    for i in range(N):
        dx, dy = rng.normal(0, perturb, 2)
        x, y = project(points[i,0]+dx, points[i,1]+dy)
        points[i] = [x, y]
    
    areas = compute_all_areas(points)
    cur_min = np.min(areas)
    best_min = cur_min
    best_pts = points.copy()
    
    # Very gentle annealing with small steps
    n_phases = 4
    phase_iters = total_iters // n_phases
    
    for phase in range(n_phases):
        if phase > 0:
            points = best_pts.copy()
            for i in range(N):
                dx, dy = rng.normal(0, 0.003, 2)
                x, y = project(points[i,0]+dx, points[i,1]+dy)
                points[i] = [x, y]
            areas = compute_all_areas(points)
            cur_min = np.min(areas)
        
        T0 = 0.0005 / (1 + phase*0.5)
        T_end = 1e-10
        cool = (T_end / T0) ** (1.0 / phase_iters)
        T = T0
        
        for it in range(phase_iters):
            T *= cool
            step = 0.01 * (T/T0)**0.25 + 0.0001
            
            if rng.random() < 0.5:
                min_tri = np.argmin(areas)
                tri = TRIPLETS[min_tri]
                idx = tri[rng.randint(3)]
                step *= 1.2
            else:
                idx = rng.randint(N)
            
            old = points[idx].copy()
            nx, ny = project(old[0] + rng.normal(0, step), old[1] + rng.normal(0, step))
            points[idx] = [nx, ny]
            
            new_areas = update_areas(points, idx, areas)
            new_min = np.min(new_areas)
            
            delta = new_min - cur_min
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-30)):
                areas = new_areas
                cur_min = new_min
                if cur_min > best_min:
                    best_min = cur_min
                    best_pts = points.copy()
            else:
                points[idx] = old
    
    return best_min / TRI_AREA, best_pts


def run_seed(seed):
    t0 = time.time()
    metric, pts = refine_sa(seed)
    return {'seed': seed, 'metric': metric, 'points': pts.tolist(), 'time': time.time()-t0}


def main():
    seeds = list(range(10))
    print(f"Refining best config with {len(seeds)} runs (10M iters each)...")
    t0 = time.time()
    
    with mp.Pool(min(mp.cpu_count(), len(seeds))) as pool:
        results = pool.map(run_seed, seeds)
    
    results.sort(key=lambda r: r['metric'], reverse=True)
    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"\nAll results:")
    for r in results:
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  time={r['time']:.1f}s")
    
    best = results[0]
    print(f"\nBest: metric={best['metric']:.10f}")
    print("Points:")
    for x, y in best['points']:
        print(f"  [{x:.16f}, {y:.16f}],")
    
    with open('orbits/known-literature/best_config.json', 'w') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    main()
