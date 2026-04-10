import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations

mpl.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.pad_inches': 0.2,
})

SQRT3 = np.sqrt(3.0)
TRI_AREA = 0.5 * SQRT3 / 2.0
TRIPLETS = np.array(list(combinations(range(11), 3)), dtype=np.int32)

PARENT = np.array([
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

BEST = np.array([
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

def compute_all_areas(points):
    p = points[TRIPLETS]
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    return 0.5 * np.abs(
        a[:, 0] * (b[:, 1] - c[:, 1]) +
        b[:, 0] * (c[:, 1] - a[:, 1]) +
        c[:, 0] * (a[:, 1] - b[:, 1])
    )

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

# Panel (a): Point configurations
ax = axes[0]
tri_verts = np.array([[0,0], [1,0], [0.5, SQRT3/2], [0,0]])
ax.plot(tri_verts[:,0], tri_verts[:,1], 'k-', lw=1.5)
ax.scatter(PARENT[:,0], PARENT[:,1], c='#4477AA', s=60, zorder=5, label='Parent (0.03630)', alpha=0.7)
ax.scatter(BEST[:,0], BEST[:,1], c='#EE6677', s=60, zorder=5, label='Best (0.03653)', marker='D', alpha=0.9)
for i in range(11):
    ax.annotate('', xy=BEST[i], xytext=PARENT[i],
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8))
ax.set_title('(a) Point configurations', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
ax.legend(loc='upper right', fontsize=10, frameon=False)

# Panel (b): Area distribution comparison
ax = axes[1]
areas_parent = np.sort(compute_all_areas(PARENT)) / TRI_AREA
areas_best = np.sort(compute_all_areas(BEST)) / TRI_AREA
x_idx = np.arange(165)
ax.plot(x_idx[:20], areas_parent[:20], 'o-', color='#4477AA', lw=1.5, ms=4, label='Parent', alpha=0.8)
ax.plot(x_idx[:20], areas_best[:20], 'D-', color='#EE6677', lw=1.5, ms=4, label='Best', alpha=0.8)
ax.axhline(y=0.03653, color='#228833', ls='--', lw=1, label='SOTA', alpha=0.7)
ax.set_title('(b) Smallest 20 triangle areas', fontweight='bold')
ax.set_xlabel('Triangle rank (sorted)')
ax.set_ylabel('Normalized area')
ax.legend(frameon=False)

# Panel (c): Optimization trajectory
ax = axes[2]
rounds = ['Parent', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'SA1', 'SA2', 'SA3']
metrics = [0.036301, 0.036302, 0.036320, 0.036323, 0.036343, 0.036389, 0.036429, 0.036528, 0.036530, 0.036530]
colors = ['#4477AA'] + ['#CCBB44']*6 + ['#EE6677']*3
ax.bar(range(len(rounds)), metrics, color=colors, edgecolor='#333333', lw=0.5)
ax.axhline(y=0.03653, color='#228833', ls='--', lw=1.5, label='SOTA (0.03653)')
ax.set_xticks(range(len(rounds)))
ax.set_xticklabels(rounds, fontsize=10)
ax.set_ylim(0.0362, 0.0366)
ax.set_title('(c) Optimization trajectory', fontweight='bold')
ax.set_ylabel('Normalized min area')
ax.legend(frameon=False, loc='lower right')

fig.suptitle('Gradient-Local Optimization for Heilbronn Triangle (n=11)', fontsize=18, fontweight='bold', y=1.02)
fig.savefig('orbits/gradient-local/figures/results.png', bbox_inches='tight')
plt.close(fig)
print("Figure saved")
