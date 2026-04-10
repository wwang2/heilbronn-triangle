"""Generate visualization of the best point configuration."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import sys
import os

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    
    
})

sys.path.insert(0, os.path.dirname(__file__))
from solution import BEST_POINTS

SQRT3 = np.sqrt(3)
TRI_VERTS = np.array([[0, 0], [1, 0], [0.5, SQRT3/2], [0, 0]])
EQUILATERAL_AREA = 0.5 * SQRT3 / 2


def compute_all_areas(points):
    idx = np.array(list(itertools.combinations(range(len(points)), 3)))
    a, b, c = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    ), idx


def main():
    points = BEST_POINTS
    if points is None:
        print("No best points found in solution.py")
        return
    
    areas, triplet_idx = compute_all_areas(points)
    min_area = areas.min()
    min_area_norm = min_area / EQUILATERAL_AREA
    
    # Find the smallest triplets
    sorted_idx = np.argsort(areas)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    
    # Panel (a): Point configuration
    ax = axes[0]
    ax.plot(TRI_VERTS[:, 0], TRI_VERTS[:, 1], 'k-', linewidth=1.5)
    ax.scatter(points[:, 0], points[:, 1], c='#2196F3', s=80, zorder=5, edgecolors='black', linewidth=0.8)
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_title(f'(a) Point configuration (n=11)', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3/2 + 0.05)
    
    # Panel (b): Smallest triangles highlighted
    ax = axes[1]
    ax.plot(TRI_VERTS[:, 0], TRI_VERTS[:, 1], 'k-', linewidth=1.5)
    ax.scatter(points[:, 0], points[:, 1], c='#2196F3', s=60, zorder=5, edgecolors='black', linewidth=0.8)
    
    # Highlight 5 smallest triangles
    colors = ['#FF5252', '#FF9800', '#FFEB3B', '#8BC34A', '#00BCD4']
    for rank, ci in enumerate(sorted_idx[:5]):
        tri = triplet_idx[ci]
        tri_pts = points[tri]
        triangle = plt.Polygon(tri_pts, fill=True, facecolor=colors[rank], 
                              edgecolor=colors[rank], alpha=0.35, linewidth=1.5)
        ax.add_patch(triangle)
        area_norm = areas[ci] / EQUILATERAL_AREA
        cx, cy = tri_pts.mean(axis=0)
        ax.annotate(f'{area_norm:.5f}', (cx, cy), fontsize=7, ha='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_title(f'(b) 5 smallest triangles', fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, SQRT3/2 + 0.05)
    
    # Panel (c): Area distribution
    ax = axes[2]
    ax.hist(areas / EQUILATERAL_AREA, bins=50, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(min_area_norm, color='#FF5252', linewidth=2, linestyle='--', 
               label=f'min = {min_area_norm:.5f}')
    ax.axvline(0.03653, color='#4CAF50', linewidth=2, linestyle=':', 
               label=f'SOTA = 0.03653')
    ax.set_title('(c) Triangle area distribution', fontweight='bold')
    ax.set_xlabel('Normalized area')
    ax.set_ylabel('Count')
    ax.legend(frameon=False)
    
    fig.suptitle(f'Heilbronn Triangle (n=11) — SA Result: {min_area_norm:.5f} ({min_area_norm/0.03653*100:.1f}% of SOTA)',
                fontsize=16, fontweight='bold', y=1.02)
    
    outpath = os.path.join(os.path.dirname(__file__), 'figures', 'results.png')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved {outpath}")


if __name__ == '__main__':
    main()
