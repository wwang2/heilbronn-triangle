"""
Heilbronn triangle problem: place 11 points in equilateral triangle to maximize
the minimum triangle area among all C(11,3)=165 triplets.

Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)

Best configuration found via intensive Numba-JIT simulated annealing.
Metric: 0.036530 (99.999% of SOTA 0.03653)
"""
import numpy as np


def heilbronn_triangle11() -> np.ndarray:
    """Return the best known 11-point configuration.
    
    Found by: 8-phase Numba-JIT SA, 800M iterations, seed=42, round 2
    Parent: known-literature (metric=0.036301)
    """
    points = np.array([
        [0.1062293556864848, 0.0710758645911361],
        [0.8521738851544579, 0.2560423324049147],
        [0.4999923178324033, 0.2111247288075353],
        [0.2774464405065116, 0.0000000295515091],
        [0.1478227903234471, 0.2560365776868428],
        [0.4279868509264755, 0.7412948545022972],
        [0.5720190466372811, 0.7412846410657106],
        [0.8937681931796337, 0.0710758658108138],
        [0.4093311687853985, 0.4392800602323884],
        [0.7225448801260982, 0.0000000310705539],
        [0.5906639243724431, 0.4392968836889751],
    ])
    return points
