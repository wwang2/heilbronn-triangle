"""
Heilbronn triangle problem: place 11 points in equilateral triangle to maximize
the minimum triangle area among all C(11,3)=165 triplets.

Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)

Best configuration found via simulated annealing optimization.
"""
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Return the best known 11-point configuration.
    
    Metric: 0.0363 (normalized min area / equilateral triangle area)
    Found by: multi-start simulated annealing with 3M iterations, 20 seeds
    """
    points = np.array([
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
    return points
