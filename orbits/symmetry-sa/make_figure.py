"""Regenerate symmetry_analysis.png with larger fonts and fixed labels."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import combinations

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
    'savefig.pad_inches': 0.25,
})

# Best 11-point configuration
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

# Mirror pairs: (left_idx, right_idx) - left has x < 0.5
pairs = [(0, 7), (4, 1), (3, 9), (5, 6), (8, 10)]
axis_pt = 2  # point 2 is on x=0.5

# Pair colors
pair_colors = ['tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown']

# Unit equilateral triangle vertices
tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

def triangle_area(a, b, c):
    return abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])) / 2.0

# Compute all triangle areas
n = len(points)
areas = []
for i, j, k in combinations(range(n), 3):
    areas.append(triangle_area(points[i], points[j], points[k]))
areas = sorted(areas)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# ---- Panel (a): Configuration with mirror symmetry ----
ax = axes[0]
ax.set_title('(a) Configuration with mirror symmetry', fontweight='bold')

# Draw triangle
tri_patch = plt.Polygon(tri, fill=False, edgecolor='black', linewidth=1.5)
ax.add_patch(tri_patch)

# Draw symmetry axis (vertical red dashed line at x=0.5)
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.2, label='Symmetry axis (x=0.5)')

# Draw mirror pair connections and points
for cidx, (li, ri) in enumerate(pairs):
    lp = points[li]
    rp = points[ri]
    ax.plot([lp[0], rp[0]], [lp[1], rp[1]], color=pair_colors[cidx],
            linestyle='-', linewidth=0.8, alpha=0.5)
    ax.scatter(*lp, color=pair_colors[cidx], s=80, zorder=5)
    ax.scatter(*rp, color=pair_colors[cidx], s=80, zorder=5)

# Axis point
ax.scatter(*points[axis_pt], color='gold', s=120, marker='s', zorder=6, label='Axis point')

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 0.92)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(loc='upper center', fontsize=11, frameon=False)
ax.set_aspect('equal')

# ---- Panel (b): Smallest 25 triangle areas ----
ax = axes[1]
ax.set_title('(b) Smallest 25 triangle areas', fontweight='bold')

top25 = areas[:25]
x_pos = np.arange(1, 26)
bar_colors = ['tab:blue' if i < 10 else 'steelblue' for i in range(25)]
ax.bar(x_pos, top25, color=bar_colors, edgecolor='white', linewidth=0.3)

sota = 0.03653
ax.axhline(y=sota, color='crimson', linestyle='--', linewidth=1.5, label=f'SOTA benchmark ({sota})')
ax.set_xlabel('Triangle rank (smallest first)')
ax.set_ylabel('Normalized area')
ax.legend(loc='upper left', fontsize=11, frameon=False)
ax.set_xticks([1, 5, 10, 15, 20, 25])

# Annotate the 10 tightest
ax.annotate('10 tightest\ntriangles', xy=(5, areas[4]), xytext=(12, areas[0]*0.997),
            fontsize=11, arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            ha='center')

# ---- Panel (c): Pair distances from axis ----
ax = axes[2]
ax.set_title('(c) Pair distances from axis', fontweight='bold')

# Draw triangle
tri_patch2 = plt.Polygon(tri, fill=False, edgecolor='black', linewidth=1.5)
ax.add_patch(tri_patch2)

# Draw symmetry axis
ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.2)

# For each pair, draw a horizontal double-arrow showing distance from axis
# Stagger label positions to avoid overlap
label_offsets = [0.03, 0.03, 0.03, 0.03, 0.03]  # y offset for text

for cidx, (li, ri) in enumerate(pairs):
    lp = points[li]
    rp = points[ri]
    dist = rp[0] - 0.5  # half-distance from axis

    # Draw the pair points
    ax.scatter(*lp, color=pair_colors[cidx], s=80, zorder=5)
    ax.scatter(*rp, color=pair_colors[cidx], s=80, zorder=5)

    # Draw a horizontal arrow from axis to right point
    y_arr = lp[1]
    ax.annotate('', xy=(rp[0], y_arr), xytext=(0.5, y_arr),
                arrowprops=dict(arrowstyle='<->', color=pair_colors[cidx], lw=1.5))

    # Label: place to the right, with slight vertical stagger to avoid overlaps
    # Determine if label would overlap with others by checking y proximity
    text_y = y_arr
    # Place label at right side, offset text slightly above/below if y is close to another
    ax.text(rp[0] + 0.02, text_y, f'd={dist:.3f}',
            color=pair_colors[cidx], fontsize=10, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))

# Axis point
ax.scatter(*points[axis_pt], color='gold', s=120, marker='s', zorder=6)

ax.set_xlim(-0.05, 1.22)
ax.set_ylim(-0.05, 0.92)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')

# Add pair legend outside
legend_handles = [mpatches.Patch(color=pair_colors[i], label=f'Pair {pairs[i]}') for i in range(5)]
legend_handles.append(mpatches.Patch(color='gold', label=f'Axis pt ({axis_pt})'))
axes[2].legend(handles=legend_handles, loc='upper right', fontsize=9.5, frameon=True,
               framealpha=0.85, edgecolor='grey')

fig.savefig('/Users/wujiewang/code/heilbronn-triangle/.worktrees/symmetry-sa/orbits/symmetry-sa/figures/symmetry_analysis.png')
plt.close(fig)
print('Figure saved.')
