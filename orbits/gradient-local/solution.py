"""
Heilbronn triangle problem: place 11 points in equilateral triangle to maximize
the minimum triangle area among all C(11,3)=165 triplets.

Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)

Best configuration found via simulated annealing refinement.
"""
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Return the best 11-point configuration.
    
    Metric: 0.036530 (normalized min area / equilateral triangle area)
    Combined score: 0.999996 (vs SOTA 0.03653)
    Method: Simulated annealing from gradient-optimized starting point
    """
    points = np.array([
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
    return points
