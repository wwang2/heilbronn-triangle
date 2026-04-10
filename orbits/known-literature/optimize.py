"""
Multi-start optimization for the Heilbronn triangle problem with n=11.

Uses scipy.optimize.minimize with the negative of min-triangle-area as objective.
Multiple random restarts + perturbation of best known solutions.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from itertools import combinations
import multiprocessing as mp
import json
import sys
import time

# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3)/2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of equilateral triangle

N = 11
# All triplet index combinations, precomputed
TRIPLETS = np.array(list(combinations(range(N), 3)))


def barycentric_to_cartesian(lam):
    """Convert barycentric coordinates (lam1, lam2) to Cartesian.
    lam3 = 1 - lam1 - lam2. Points inside triangle iff lam1,lam2 >= 0 and lam1+lam2 <= 1.
    """
    lam1, lam2 = lam
    lam3 = 1.0 - lam1 - lam2
    return lam1 * V0 + lam2 * V1 + lam3 * V2


def params_to_points(params):
    """Convert flat parameter array (22 values in [0,1]) to 11 points via barycentric coords.
    We use a sigmoid-like transform to keep points inside the triangle.
    """
    points = np.zeros((N, 2))
    for i in range(N):
        u, v = params[2*i], params[2*i+1]
        # Map (u,v) in R^2 to barycentric coords inside triangle
        # Use softmax-like transform for unconstrained optimization
        eu = np.exp(np.clip(u, -20, 20))
        ev = np.exp(np.clip(v, -20, 20))
        e0 = 1.0  # third barycentric coordinate
        s = eu + ev + e0
        lam1, lam2 = eu/s, ev/s
        points[i] = lam1 * V0 + lam2 * V1 + (1 - lam1 - lam2) * V2
    return points


def min_triangle_area_from_points(points):
    """Compute minimum triangle area among all C(11,3) triplets. Vectorized."""
    p = points[TRIPLETS]  # shape (165, 3, 2)
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    areas = 0.5 * np.abs(a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1]))
    return np.min(areas)


def objective(params):
    """Negative of min triangle area (we minimize this)."""
    points = params_to_points(params)
    return -min_triangle_area_from_points(points)


def smooth_objective(params, alpha=0.01):
    """Smooth approximation using log-sum-exp of negative areas."""
    points = params_to_points(params)
    p = points[TRIPLETS]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    areas = 0.5 * np.abs(a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1]))
    # Smooth min approximation: -alpha * log(sum(exp(-areas/alpha)))
    shifted = -areas / alpha
    max_shifted = np.max(shifted)
    smooth_min = -alpha * (max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))))
    return smooth_min  # this is approximately -min(areas)


def random_init(rng):
    """Generate random initial parameters."""
    return rng.standard_normal(2 * N)


def points_to_params(points):
    """Convert points back to parameter space (inverse of params_to_points)."""
    params = np.zeros(2 * N)
    for i in range(N):
        x, y = points[i]
        # Solve for barycentric coords
        # p = lam1*V0 + lam2*V1 + (1-lam1-lam2)*V2
        # x = lam1*0 + lam2*1 + (1-lam1-lam2)*0.5 = lam2 + 0.5 - 0.5*lam1 - 0.5*lam2
        # x = 0.5 - 0.5*lam1 + 0.5*lam2
        # y = lam1*0 + lam2*0 + (1-lam1-lam2)*sqrt(3)/2 = (1-lam1-lam2)*sqrt(3)/2
        h = np.sqrt(3)/2
        lam12 = 1.0 - y/h  # lam1 + lam2
        lam2_minus_lam1 = 2*(x - 0.5)  # from x equation
        lam1 = (lam12 - lam2_minus_lam1) / 2
        lam2 = (lam12 + lam2_minus_lam1) / 2
        lam3 = 1.0 - lam1 - lam2
        # Clamp
        lam1 = np.clip(lam1, 1e-8, 1-2e-8)
        lam2 = np.clip(lam2, 1e-8, 1-2e-8)
        lam3 = np.clip(lam3, 1e-8, 1-2e-8)
        # Inverse softmax: u = log(lam1/lam3), v = log(lam2/lam3)
        params[2*i] = np.log(lam1 / lam3)
        params[2*i+1] = np.log(lam2 / lam3)
    return params


def optimize_single(seed):
    """Run one optimization from a random start. Returns (metric, points)."""
    rng = np.random.RandomState(seed)
    
    best_metric = 0.0
    best_points = None
    
    # Try multiple strategies
    for attempt in range(5):
        if attempt == 0:
            # Pure random
            x0 = rng.standard_normal(2*N) * 0.5
        elif attempt == 1:
            # Start from a spread-out configuration
            # Place points roughly evenly: vertices + edge midpoints + interior
            pts = np.array([
                V0, V1, V2,
                (V0+V1)/2, (V1+V2)/2, (V0+V2)/2,
                (V0+V1+V2)/3,  # centroid
                V0*0.6 + V1*0.2 + V2*0.2,
                V0*0.2 + V1*0.6 + V2*0.2,
                V0*0.2 + V1*0.2 + V2*0.6,
                (V0+V1)*0.3 + V2*0.4,
            ])
            x0 = points_to_params(pts) + rng.standard_normal(2*N) * 0.1
        elif attempt == 2:
            # Perturbed vertices + thirds
            pts = np.array([
                V0, V1, V2,
                V0*2/3 + V1*1/3, V0*1/3 + V1*2/3,
                V1*2/3 + V2*1/3, V1*1/3 + V2*2/3,
                V0*2/3 + V2*1/3, V0*1/3 + V2*2/3,
                (V0+V1+V2)/3 + np.array([0.05, 0]),
                (V0+V1+V2)/3 + np.array([-0.05, 0]),
            ])
            x0 = points_to_params(pts) + rng.standard_normal(2*N) * 0.15
        else:
            # Random with different scale
            x0 = rng.standard_normal(2*N) * (0.3 + 0.5 * attempt)
        
        # First pass: smooth objective for gradient-friendly landscape
        res = minimize(smooth_objective, x0, args=(0.005,), method='L-BFGS-B',
                      options={'maxiter': 2000, 'ftol': 1e-15})
        
        # Second pass: refine with true objective using Nelder-Mead
        res2 = minimize(objective, res.x, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-15, 'adaptive': True})
        
        pts = params_to_points(res2.x)
        metric = min_triangle_area_from_points(pts) / TRI_AREA
        
        if metric > best_metric:
            best_metric = metric
            best_points = pts.copy()
    
    return best_metric, best_points


def run_with_seed(seed):
    """Wrapper for multiprocessing."""
    t0 = time.time()
    metric, points = optimize_single(seed)
    elapsed = time.time() - t0
    return {'seed': seed, 'metric': metric, 'points': points.tolist(), 'time': elapsed}


def main():
    seeds = list(range(50))  # 50 different random starts
    
    print(f"Running {len(seeds)} optimization seeds in parallel...")
    t0 = time.time()
    
    with mp.Pool(min(mp.cpu_count(), len(seeds))) as pool:
        results = pool.map(run_with_seed, seeds)
    
    total_time = time.time() - t0
    
    # Sort by metric descending
    results.sort(key=lambda r: r['metric'], reverse=True)
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"\nTop 5 results:")
    for r in results[:5]:
        print(f"  seed={r['seed']:3d}  metric={r['metric']:.10f}  time={r['time']:.1f}s")
    
    best = results[0]
    print(f"\nBest: seed={best['seed']}  metric={best['metric']:.10f}")
    print(f"Best points:")
    for i, (x, y) in enumerate(best['points']):
        print(f"  [{x:.16f}, {y:.16f}],")
    
    # Save best points for solution.py
    with open('orbits/known-literature/best_config.json', 'w') as f:
        json.dump({'metric': best['metric'], 'points': best['points'], 
                   'seed': best['seed']}, f, indent=2)
    
    return best


if __name__ == '__main__':
    main()
