"""
Heilbronn Triangle Problem (n=11) — Simulated Annealing with Random Restarts.

Places 11 points in an equilateral triangle to maximize the minimum triangle
area among all C(11,3)=165 triplets.
"""

import numpy as np
import itertools

# Best coordinates found by SA optimization (will be updated after running optimize.py)
BEST_POINTS = np.array([
    [0.680644513236530, 0.052555235191888],
    [0.656847102488838, 0.411596375185949],
    [0.166605121248576, 0.288517563027240],
    [0.091358205660673, 0.052582370509936],
    [0.434547423123960, 0.438308651720954],
    [0.884995700385480, 0.199193290021223],
    [0.947309749475442, 0.000000000000000],
    [0.205850610636699, 0.000000000000000],
    [0.545735169414415, 0.786809766665901],
    [0.460137759869902, 0.796981978575598],
    [0.504114885614634, 0.203452524705692]
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
