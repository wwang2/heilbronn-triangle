"""
Gradient-based local optimization for Heilbronn triangle problem (n=11).
Uses only numpy (no scipy). Implements:
1. Numerical-gradient ascent with softmin smoothing + beta continuation
2. Nelder-Mead (hand-coded)
3. Coordinate-wise golden section search
All starting from the parent configuration.
"""
import numpy as np
from itertools import combinations
from multiprocessing import Pool
import time
import json
import os

SQRT3 = np.sqrt(3.0)
HALF_SQRT3 = SQRT3 / 2.0
TRI_AREA = 0.5 * 1.0 * HALF_SQRT3

TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=np.int32)

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
    """Vectorized computation of all 165 triangle areas."""
    p = points[TRIPLETS]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return areas


def evaluate_config(points):
    areas = compute_all_areas(points)
    return np.min(areas) / TRI_AREA


def project_to_triangle(points):
    """Project points into the equilateral triangle."""
    pts = points.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(y, 0.0)
        if y > SQRT3 * x:
            xp = (x + y / SQRT3) / 2.0
            yp = SQRT3 * xp
            x, y = xp, yp
        if y > SQRT3 * (1 - x):
            xp = (x - y / SQRT3 + 1) / 2.0
            yp = SQRT3 * (1 - xp)
            x, y = xp, yp
        y = max(y, 0.0)
        pts[i] = [x, y]
    return pts


def softmin_value(points, beta):
    """Log-sum-exp softmin of triangle areas. Returns the smooth min approximation."""
    areas = compute_all_areas(points)
    scaled = -beta * areas
    max_s = np.max(scaled)
    return -(max_s + np.log(np.sum(np.exp(scaled - max_s)))) / beta


def numerical_gradient(points, beta, h=1e-8):
    """Compute gradient of softmin w.r.t. point coordinates via finite differences."""
    grad = np.zeros_like(points)
    f0 = softmin_value(points, beta)
    for i in range(11):
        for j in range(2):
            pts_plus = points.copy()
            pts_plus[i, j] += h
            grad[i, j] = (softmin_value(pts_plus, beta) - f0) / h
    return grad


def check_feasibility(points, tol=1e-9):
    for x, y in points:
        if y < -tol or y > SQRT3 * x + tol or y > SQRT3 * (1 - x) + tol:
            return False
    return True


# ============================================================
# Method 1: Gradient ascent with beta continuation
# ============================================================
def run_gradient_continuation(seed, perturbation_scale=1e-4):
    np.random.seed(seed)
    start_time = time.time()

    pts = PARENT_POINTS.copy() + np.random.randn(11, 2) * perturbation_scale
    pts = project_to_triangle(pts)
    best_points = PARENT_POINTS.copy()
    best_metric = evaluate_config(best_points)

    beta_schedule = [10, 30, 100, 300, 1000, 3000, 10000]

    for beta in beta_schedule:
        lr = min(1e-4, 1.0 / beta)  # learning rate scales inversely with beta
        patience = 0
        prev_val = softmin_value(pts, beta)

        for step in range(2000):
            grad = numerical_gradient(pts, beta)
            # Gradient ascent (maximize softmin)
            new_pts = pts + lr * grad
            new_pts = project_to_triangle(new_pts)
            new_val = softmin_value(new_pts, beta)

            if new_val > prev_val:
                pts = new_pts
                prev_val = new_val
                patience = 0
                # Try increasing lr
                lr *= 1.05
            else:
                patience += 1
                lr *= 0.5
                if patience > 20:
                    break

            if check_feasibility(pts):
                metric = evaluate_config(pts)
                if metric > best_metric:
                    best_metric = metric
                    best_points = pts.copy()

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'grad-cont'}


# ============================================================
# Method 2: Nelder-Mead (hand-coded)
# ============================================================
def run_nelder_mead(seed, perturbation_scale=1e-4):
    np.random.seed(seed)
    start_time = time.time()

    best_points = PARENT_POINTS.copy()
    best_metric = evaluate_config(best_points)
    n = 22  # 11 points * 2 coords

    def objective(x):
        pts = project_to_triangle(x.reshape(11, 2))
        return -evaluate_config(pts)

    # Initialize simplex
    x0 = PARENT_POINTS.flatten() + np.random.randn(n) * perturbation_scale
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1, i] += perturbation_scale

    fvals = np.array([objective(s) for s in simplex])

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

    for iteration in range(100000):
        # Sort
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]

        # Check convergence
        if np.max(np.abs(fvals - fvals[0])) < 1e-16 and iteration > 1000:
            break

        # Centroid (excluding worst)
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = objective(xr)

        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            if fr < fvals[-1]:
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                fc = objective(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])
            else:
                # Inside contraction
                xc = centroid - rho * (centroid - simplex[-1])
                fc = objective(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = objective(simplex[i])

        # Track best
        best_x = simplex[np.argmin(fvals)]
        candidate = project_to_triangle(best_x.reshape(11, 2))
        if check_feasibility(candidate):
            metric = evaluate_config(candidate)
            if metric > best_metric:
                best_metric = metric
                best_points = candidate.copy()

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'nelder-mead'}


# ============================================================
# Method 3: Coordinate descent with golden section
# ============================================================
def run_coordinate_descent(seed, perturbation_scale=1e-4):
    np.random.seed(seed)
    start_time = time.time()

    pts = PARENT_POINTS.copy() + np.random.randn(11, 2) * perturbation_scale
    pts = project_to_triangle(pts)
    best_points = PARENT_POINTS.copy()
    best_metric = evaluate_config(best_points)

    phi = (np.sqrt(5) - 1) / 2

    for outer in range(50):
        improved = False
        # Randomize coordinate order
        coords = list(range(22))
        np.random.shuffle(coords)

        for c in coords:
            i, j = c // 2, c % 2
            original = pts[i, j]

            # Search range
            delta = 0.01 * (0.95 ** outer)
            lo = original - delta
            hi = original + delta

            # Golden section search to maximize min_area
            for _ in range(40):
                x1 = hi - phi * (hi - lo)
                x2 = lo + phi * (hi - lo)

                pts[i, j] = x1
                pts_proj1 = project_to_triangle(pts.copy())
                f1 = evaluate_config(pts_proj1)

                pts[i, j] = x2
                pts_proj2 = project_to_triangle(pts.copy())
                f2 = evaluate_config(pts_proj2)

                if f1 > f2:
                    hi = x2
                else:
                    lo = x1

            best_val = (lo + hi) / 2
            pts[i, j] = best_val
            pts = project_to_triangle(pts)

            if check_feasibility(pts):
                metric = evaluate_config(pts)
                if metric > best_metric:
                    best_metric = metric
                    best_points = pts.copy()
                    improved = True

        if not improved:
            break

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'coord-descent'}


# ============================================================
# Method 4: Stochastic perturbation + local search (basin hopping style)
# ============================================================
def run_basin_hopping(seed, perturbation_scale=1e-3):
    np.random.seed(seed)
    start_time = time.time()

    pts = PARENT_POINTS.copy()
    best_points = PARENT_POINTS.copy()
    best_metric = evaluate_config(best_points)

    for hop in range(200):
        # Perturb
        scale = perturbation_scale * (0.995 ** hop)
        trial = pts + np.random.randn(11, 2) * scale
        trial = project_to_triangle(trial)

        # Local refinement: gradient ascent with high beta
        beta = 5000
        lr = 1e-5
        for step in range(500):
            grad = numerical_gradient(trial, beta, h=1e-9)
            new_trial = trial + lr * grad
            new_trial = project_to_triangle(new_trial)
            if softmin_value(new_trial, beta) > softmin_value(trial, beta):
                trial = new_trial
                lr *= 1.02
            else:
                lr *= 0.5
                if lr < 1e-12:
                    break

        if check_feasibility(trial):
            metric = evaluate_config(trial)
            if metric > best_metric:
                best_metric = metric
                best_points = trial.copy()
                pts = trial.copy()

    elapsed = time.time() - start_time
    return {'seed': seed, 'metric': best_metric, 'points': best_points, 'time': elapsed, 'method': 'basin-hop'}


def run_single_config(args):
    method, seed = args
    try:
        if method == 'grad-cont':
            return run_gradient_continuation(seed)
        elif method == 'nelder-mead':
            return run_nelder_mead(seed)
        elif method == 'coord-descent':
            return run_coordinate_descent(seed)
        elif method == 'basin-hop':
            return run_basin_hopping(seed)
    except Exception as e:
        return {'seed': seed, 'metric': evaluate_config(PARENT_POINTS), 'points': PARENT_POINTS, 'time': 0, 'method': method, 'error': str(e)}


if __name__ == '__main__':
    parent_metric = evaluate_config(PARENT_POINTS)
    print(f"Parent metric: {parent_metric:.10f}")
    print(f"SOTA:          0.03653")
    print()

    configs = [
        ('grad-cont', 42), ('grad-cont', 123), ('grad-cont', 7),
        ('nelder-mead', 42), ('nelder-mead', 123), ('nelder-mead', 7),
        ('coord-descent', 42), ('coord-descent', 123), ('coord-descent', 7),
        ('basin-hop', 42), ('basin-hop', 123), ('basin-hop', 7),
    ]

    print(f"Running {len(configs)} optimization configs in parallel...")
    with Pool(min(len(configs), 8)) as pool:
        results = pool.map(run_single_config, configs)

    results.sort(key=lambda r: r['metric'], reverse=True)

    print("\n=== Results (sorted by metric) ===")
    for r in results:
        err = f"  ERROR: {r.get('error')}" if 'error' in r else ""
        improvement = (r['metric'] - parent_metric) / parent_metric * 100
        print(f"  {r['method']:15s} seed={r['seed']:5d}  metric={r['metric']:.10f}  delta={improvement:+.4f}%  time={r['time']:.1f}s{err}")

    best = results[0]
    print(f"\n=== Best result ===")
    print(f"  Method: {best['method']}, Seed: {best['seed']}")
    print(f"  Metric: {best['metric']:.10f}")
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
