"""Differential evolution for Heilbronn triangle n=11."""
import numpy as np
from itertools import combinations
from scipy.optimize import differential_evolution
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2
TRI_AREA = H / 2
TRIPLETS = np.array(list(combinations(range(N), 3)))
I0, I1, I2 = TRIPLETS[:, 0], TRIPLETS[:, 1], TRIPLETS[:, 2]

def params_to_points(params):
    points = np.zeros((N, 2))
    for i in range(N):
        u, v = params[2*i], params[2*i+1]
        if u + v > 1:
            u, v = 1-v, 1-u
        points[i, 0] = u * 0 + v * 1 + (1-u-v) * 0.5
        points[i, 1] = (1-u-v) * H
    return points

def objective(params):
    points = params_to_points(params)
    a, b, c = points[I0], points[I1], points[I2]
    areas = 0.5 * np.abs(
        a[:, 0]*(b[:, 1]-c[:, 1]) + b[:, 0]*(c[:, 1]-a[:, 1]) + c[:, 0]*(a[:, 1]-b[:, 1])
    )
    return -np.min(areas) / TRI_AREA

bounds = [(0, 1)] * (2 * N)

t0 = time.time()
result = differential_evolution(objective, bounds, seed=42, maxiter=3000, 
                                popsize=60, tol=1e-14, 
                                mutation=(0.5, 1.5), recombination=0.9,
                                workers=-1, updating='deferred')
elapsed = time.time() - t0
print(f"DE result: {-result.fun:.10f} in {elapsed:.1f}s")
pts = params_to_points(result.x)
print("Points:")
for x, y in pts:
    print(f"  [{x:.16f}, {y:.16f}],")
