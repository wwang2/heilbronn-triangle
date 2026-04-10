"""
Heilbronn triangle problem: place 11 points in equilateral triangle to maximize
the minimum triangle area among all C(11,3)=165 triplets.

Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)

Best configuration found via iterated targeted Nelder-Mead optimization
with perturbed restarts from the known-literature parent configuration.
"""
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Return the best 11-point configuration found by gradient-local optimization.
    
    Metric: 0.036429 (normalized min area / equilateral triangle area)
    Method: Iterated targeted Nelder-Mead with beta-continuation restarts
    """
    points = np.array([
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
    return points
