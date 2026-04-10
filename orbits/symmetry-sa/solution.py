import numpy as np

def heilbronn_triangle11() -> np.ndarray:
    """Best configuration found via symmetry-enforced optimization.
    
    The optimal 11-point Heilbronn configuration has exact mirror symmetry
    about x=0.5. Five pairs of points are related by (x,y) <-> (1-x,y),
    plus one point on the axis at x=0.5.
    
    Pairs: (0,7), (1,4), (3,9), (5,6), (8,10). Axis: point 2.
    
    Found by: symmetrized coordinate descent from Numba-JIT SA result.
    Metric: 0.036530 (99.9999% of SOTA 0.03653)
    """
    points = np.array([
        [0.1062306703654052, 0.0710761657400959],
        [0.8521750796907640, 0.2560402760808697],
        [0.5000000000000000, 0.2111254344044703],
        [0.2774515797949464, 0.0000000000000000],
        [0.1478249203092360, 0.2560402760808697],
        [0.4279840016586173, 0.7412900423154831],
        [0.5720159983413827, 0.7412900423154831],
        [0.8937693296345948, 0.0710761657400959],
        [0.4093343198961193, 0.4392897750602333],
        [0.7225484202050536, 0.0000000000000000],
        [0.5906656801038807, 0.4392897750602333],
    ])
    return points
