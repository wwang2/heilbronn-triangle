"""
Heilbronn Triangle Problem (n=11) — Static lookup of best configuration.

This module does NOT run any optimization at import or call time. It simply
returns the hardcoded best point-set discovered by optimize_v4.py (Basin-hopping
+ Differential Evolution + local Nelder-Mead search). The optimization itself
is in optimize_v4.py; this file records its output for deterministic evaluation.
"""

import numpy as np

BEST_POINTS = np.array([
    [0.571410114990977, 0.160903132947647],
    [0.967840603269951, 0.055701709077209],
    [0.834572069630165, 0.000007622012137],
    [0.634431916255954, 0.633182494670282],
    [0.433427347279021, 0.359597997163961],
    [0.118473366889313, 0.205201890796039],
    [0.098922062316121, 0.055790652902838],
    [0.378869492469816, 0.656221210395555],
    [0.507327904535471, 0.763089164415980],
    [0.293006043913453, 0.000051847694994],
    [0.657443788714565, 0.321098845590150],
])


def heilbronn_triangle11():
    """Return the best known 11-point configuration."""
    return BEST_POINTS.copy()
