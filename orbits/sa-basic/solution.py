"""
Heilbronn Triangle Problem (n=11) — Simulated Annealing with Random Restarts.

Places 11 points in an equilateral triangle to maximize the minimum triangle
area among all C(11,3)=165 triplets.
"""

import numpy as np
import itertools

# Best coordinates found by SA optimization (will be updated after running optimize.py)
BEST_POINTS = np.array([
    [0.734782987754903, 0.328592677642313],
    [0.495425141606417, 0.363353151139192],
    [0.136555323097653, 0.077931197925199],
    [0.311748332974034, 0.539963951885926],
    [0.665735849265812, 0.578962492220475],
    [0.513620422531096, 0.713599299123511],
    [0.564497014791776, 0.113070957650502],
    [0.710551126509691, 0.000011370569874],
    [0.130439171525644, 0.225927272379608],
    [0.341744338895933, 0.000000000000000],
    [0.954987868101435, 0.077963299405307]
])


def heilbronn_triangle11() -> np.ndarray:
    """Return the best 11-point configuration found."""
    if BEST_POINTS is not None:
        return BEST_POINTS.copy()
    # Fallback: run a quick optimization
    result = run_sa_optimization(n_restarts=5, max_iter=50000, seed=42)
    return result['points']


def triangle_area_all(points: np.ndarray) -> np.ndarray:
    """Vectorized computation of all C(n,3) triangle areas."""
    n = len(points)
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )
    return areas


def min_triangle_area(points: np.ndarray) -> float:
    """Compute min triangle area over all triplets."""
    return float(np.min(triangle_area_all(points)))


def random_point_in_triangle(rng):
    """Sample a random point uniformly inside the equilateral triangle."""
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    r1, r2 = rng.random(), rng.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    return v0 + r1 * (v1 - v0) + r2 * (v2 - v0)


def project_to_triangle(x, y):
    """Project point to nearest point inside the equilateral triangle."""
    sqrt3 = np.sqrt(3)
    y = max(y, 0.0)
    if y > sqrt3 * x:
        t = (x + sqrt3 * y) / 4.0
        t = max(0.0, min(0.5, t))
        x, y = t, sqrt3 * t
    if y > sqrt3 * (1 - x):
        t = (x - y / sqrt3 + 1) / 2.0
        t = max(0.5, min(1.0, t))
        x, y = t, sqrt3 * (1 - t)
    y = max(y, 0.0)
    return x, y


def simulated_annealing(n_points=11, max_iter=200000, T_start=0.1, T_end=1e-6,
                         seed=42, perturb_scale_start=0.15, perturb_scale_end=0.005):
    """
    Simulated annealing for Heilbronn triangle problem.
    Maximize min_triangle_area by exploring point configurations.
    """
    rng = np.random.RandomState(seed)
    points = np.array([random_point_in_triangle(rng) for _ in range(n_points)])

    current_obj = min_triangle_area(points)
    best_obj = current_obj
    best_points = points.copy()

    cooling_rate = (T_end / T_start) ** (1.0 / max_iter)
    scale_rate = (perturb_scale_end / perturb_scale_start) ** (1.0 / max_iter)

    T = T_start
    perturb_scale = perturb_scale_start
    no_improve_count = 0

    for iteration in range(max_iter):
        idx = rng.randint(n_points)
        old_point = points[idx].copy()

        dx = rng.randn() * perturb_scale
        dy = rng.randn() * perturb_scale
        new_x, new_y = old_point[0] + dx, old_point[1] + dy
        new_x, new_y = project_to_triangle(new_x, new_y)

        points[idx] = [new_x, new_y]
        new_obj = min_triangle_area(points)
        delta = new_obj - current_obj

        if delta > 0:
            current_obj = new_obj
            no_improve_count = 0
            if new_obj > best_obj:
                best_obj = new_obj
                best_points = points.copy()
        elif T > 0 and rng.random() < np.exp(delta / T):
            current_obj = new_obj
            no_improve_count += 1
        else:
            points[idx] = old_point
            no_improve_count += 1

        T *= cooling_rate
        perturb_scale *= scale_rate

        # Reheat if stuck
        if no_improve_count > 5000:
            T = T_start * 0.1
            perturb_scale = perturb_scale_start * 0.3
            no_improve_count = 0

    return best_obj, best_points


def run_sa_optimization(n_restarts=20, max_iter=200000, seed=0):
    """Run SA with multiple restarts and return the best result."""
    best_obj = -1
    best_points = None

    for i in range(n_restarts):
        obj, pts = simulated_annealing(
            max_iter=max_iter,
            seed=seed + i * 137,
            T_start=0.05,
            T_end=1e-7,
            perturb_scale_start=0.12,
            perturb_scale_end=0.002
        )
        if obj > best_obj:
            best_obj = obj
            best_points = pts.copy()

    equilateral_area = 0.5 * np.sqrt(3) / 2
    normalized = best_obj / equilateral_area

    return {
        'points': best_points,
        'min_area': best_obj,
        'min_area_normalized': normalized,
    }
