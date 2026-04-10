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
    Combined score: 0.999997 (vs SOTA 0.03653)
    Method: Iterated SA from gradient-optimized starting point
    """
    points = np.array([
        [0.8559678556745838, 0.0000002396801935],
        [0.6478252948064679, 0.6099843931268593],
        [0.2956488378993500, 0.0000001143876839],
        [0.4328388298412805, 0.3274481215032055],
        [0.5851015111324630, 0.1348492763781040],
        [0.5084381874446738, 0.7384883129452630],
        [0.9279835473969013, 0.1247360903895506],
        [0.3612731995118425, 0.6257435262063643],
        [0.1146692443781280, 0.0564602890110986],
        [0.6757696388215428, 0.2918845929420058],
        [0.1387253210578125, 0.2402791660411576],
    ])
    return points
