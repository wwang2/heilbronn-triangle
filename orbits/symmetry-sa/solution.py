import numpy as np

def heilbronn_triangle11() -> np.ndarray:
    """Best configuration found via symmetry-enforced Numba-JIT SA + coordinate descent.
    
    The optimal 11-point Heilbronn configuration has exact mirror symmetry
    about x=0.5. Five pairs of points are related by (x,y) <-> (1-x,y),
    plus one point on the symmetry axis at x=0.5.
    
    Pairs: (0,7), (1,4), (3,9), (5,6), (8,10). Axis: point 2.
    
    Found by: 3-phase Numba SA (symmetric-enforced -> fine-tune -> unconstrained),
    600M iterations, seed=7, followed by symmetric coordinate descent polish.
    
    Metric: 0.036530 (combined_score = 0.9999998, 99.99998% of SOTA)
    """
    points = np.array([
        [0.1062310492450264, 0.0710766377932213],
        [0.8521745046979071, 0.2560412686066662],
        [0.5000000000000000, 0.2111263091607782],
        [0.2774528419306941, 0.0000000000000000],
        [0.1478254953020929, 0.2560412686066662],
        [0.4279844421030915, 0.7412907985715750],
        [0.5720155578969085, 0.7412907985715750],
        [0.8937689507549735, 0.0710766377932213],
        [0.4093351368541814, 0.4392916413558975],
        [0.7225471580693059, 0.0000000000000000],
        [0.5906648631458187, 0.4392916413558975],
    ])
    return points
