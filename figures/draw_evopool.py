"""
EvoPool Architecture Diagram — NeurIPS paper style
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── color palette ──────────────────────────────────────────────────────────
C_POOL   = '#2563EB'   # blue   – agent pool
C_SEL    = '#0891B2'   # cyan   – selection / team
C_MAS    = '#7C3AED'   # purple – leader MAS
C_DREAM  = '#059669'   # green  – Co-Dream
C_LC     = '#EA580C'   # orange – lifecycle
C_STREAM = '#6B7280'   # gray   – task stream
C_BG     = '#F0F9FF'   # light blue bg
C_BGLC   = '#FFF7ED'   # light orange bg
C_BGDR   = '#ECFDF5'   # light green bg
C_BGMAS  = '#F5F3FF'   # light purple bg
C_TEXT   = '#111827'

def box(ax, x, y, w, h, color, bg, label, sublabel=None, radius=0.35, lw=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.05,rounding_size={radius}",
                       linewidth=lw, edgecolor=color, facecolor=bg, zorder=3)
    ax.add_patch(p)
    ty = y + h/2 + (0.15 if sublabel else 0)
    ax.text(x+w/2, ty, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color=C_TEXT, zorder=4)
    if sublabel:
        ax.text(x+w/2, y+h/2-0.25, sublabel, ha='center', va='center',
                fontsize=7.5, color='#4B5563', zorder=4, style='italic')

def arrow(ax, x1, y1, x2, y2, label='', color='#1F2937', lw=2.0, curve=0):
    style = f"arc3,rad={curve}" if curve else "arc3,rad=0"
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw, mutation_scale=18,
                                connectionstyle=style),
                zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my+0.12, label, ha='center', va='bottom',
                fontsize=7.5, color='#374151', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))

# ══════════════════════════════════════════════════════════════════════════
# 0. TITLE
# ══════════════════════════════════════════════════════════════════════════
ax.text(9, 10.55, 'EvoPool: Lifelong Multi-Agent Learning with Evolving Specialization',
        ha='center', va='center', fontsize=14, fontweight='bold', color=C_TEXT)

# ══════════════════════════════════════════════════════════════════════════
# 1. TASK STREAM  (top)
# ══════════════════════════════════════════════════════════════════════════
ax.annotate('', xy=(15.5, 9.9), xytext=(1.8, 9.9),
            arrowprops=dict(arrowstyle='->', color=C_STREAM, lw=2.5,
                            mutation_scale=20), zorder=5)
ax.text(8.5, 10.18, 'Task Stream  t = 1, 2, …, T', ha='center', va='center',
        fontsize=9, fontweight='bold', color=C_STREAM)

# domain icons
domains = [('GSM8K\n∫∑x²', 2.4), ('HotpotQA\n? ↔ ?', 4.2),
           ('MBPP\ndef f():', 6.0), ('MATH\nax²+bx', 7.8),
           ('HumanEval\n{ code }', 9.6), ('DROP\n[passage]', 11.4)]
for d_label, dx in domains:
    p = FancyBboxPatch((dx-0.55, 9.3), 1.1, 0.5,
                       boxstyle="round,pad=0.05,rounding_size=0.18",
                       linewidth=1.4, edgecolor=C_STREAM, facecolor='#F9FAFB', zorder=3)
    ax.add_patch(p)
    ax.text(dx, 9.55, d_label, ha='center', va='center',
            fontsize=6.5, color='#374151', zorder=4)

# ══════════════════════════════════════════════════════════════════════════
# 2. AGENT POOL  (left)
# ══════════════════════════════════════════════════════════════════════════
pool_bg = FancyBboxPatch((0.2, 1.2), 3.5, 7.8,
                         boxstyle="round,pad=0.1,rounding_size=0.4",
                         linewidth=2, edgecolor=C_POOL, facecolor=C_BG, zorder=1)
ax.add_patch(pool_bg)
ax.text(1.95, 8.7, 'Agent Pool  (N = 20)', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_POOL)

# draw 12 agent circles in 3×4 grid
specs = {(0,0):'Math', (1,1):'Code', (2,0):'QA'}
r_ag = 0.38
for col in range(3):
    for row in range(4):
        cx = 0.85 + col*1.05
        cy = 2.0 + row*1.6
        c = Circle((cx, cy), r_ag, facecolor='white',
                   edgecolor=C_POOL, linewidth=1.6, zorder=3)
        ax.add_patch(c)
        spec = specs.get((col, row), '')
        if spec:
            ax.text(cx, cy+0.05, spec, ha='center', va='center',
                    fontsize=6.5, fontweight='bold', color=C_POOL, zorder=4)
        else:
            ax.text(cx, cy, f'A{col*4+row+1}', ha='center', va='center',
                    fontsize=6, color='#6B7280', zorder=4)
        # mini skill bar
        for bi in range(3):
            bh = np.random.uniform(0.06, 0.18)
            bx = cx - 0.22 + bi*0.16
            ax.add_patch(plt.Rectangle((bx, cy-r_ag+0.04), 0.11, bh,
                         facecolor=C_POOL, alpha=0.55, zorder=4))

ax.text(1.95, 1.55, '⚙ profiles evolve over tasks', ha='center', va='center',
        fontsize=7, color='#6B7280', style='italic')

# ══════════════════════════════════════════════════════════════════════════
# 3. TEAM SELECTION  (center-left)
# ══════════════════════════════════════════════════════════════════════════
sel_bg = FancyBboxPatch((3.9, 4.8), 2.7, 3.2,
                        boxstyle="round,pad=0.1,rounding_size=0.35",
                        linewidth=2, edgecolor=C_SEL, facecolor='#E0F2FE', zorder=2)
ax.add_patch(sel_bg)
ax.text(5.25, 7.7, 'Team Selection', ha='center', va='center',
        fontsize=9.5, fontweight='bold', color=C_SEL)
ax.text(5.25, 7.35, 'Affinity + Diversity + Collab Score', ha='center',
        va='center', fontsize=7, color='#0369A1')

# 3 selected agents forming triangle
tri = [(5.25, 6.8), (4.3, 5.3), (6.2, 5.3)]
labels = ['Leader ★', 'Agent B', 'Agent C']
cols   = [C_MAS, C_SEL, C_SEL]
for (tx, ty), tl, tc in zip(tri, labels, cols):
    c = Circle((tx, ty), 0.42, facecolor='white',
               edgecolor=tc, linewidth=2.2, zorder=4)
    ax.add_patch(c)
    ax.text(tx, ty, tl, ha='center', va='center',
            fontsize=6.5, fontweight='bold', color=tc, zorder=5)

# thin dashed lines between selected agents
for i in range(3):
    for j in range(i+1, 3):
        x1,y1 = tri[i]; x2,y2 = tri[j]
        ax.plot([x1,x2],[y1,y2], ls='--', lw=1, color='#94A3B8', zorder=3)

# arrow: pool → selection
arrow(ax, 3.7, 5.5, 3.9, 5.5, 'select k=3', C_POOL)

# ══════════════════════════════════════════════════════════════════════════
# 4. LEADER MAS  (center)
# ══════════════════════════════════════════════════════════════════════════
mas_bg = FancyBboxPatch((6.9, 4.0), 4.1, 5.3,
                        boxstyle="round,pad=0.1,rounding_size=0.35",
                        linewidth=2, edgecolor=C_MAS, facecolor=C_BGMAS, zorder=2)
ax.add_patch(mas_bg)
ax.text(8.95, 9.0, 'Leader MAS', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_MAS)

# pipeline inside MAS
steps = ['① Decompose\n   Task', '② Assign\n   Subtasks', '③ Critique\n   Round', '④ Synthesize\n   Answer']
sy = [8.3, 7.25, 6.2, 5.1]
for i, (s, y) in enumerate(zip(steps, sy)):
    box(ax, 7.3, y-0.35, 3.3, 0.65, C_MAS, 'white', s, radius=0.2, lw=1.5)
    if i < 3:
        arrow(ax, 8.95, y-0.35, 8.95, sy[i+1]+0.3, '', C_MAS, lw=1.5)

# GenesisOnDemand note
ax.text(8.95, 4.55, '⚡ GenesisOnDemand: recruit from pool\n     if required skill not in team',
        ha='center', va='center', fontsize=6.8, color='#6D28D9',
        bbox=dict(boxstyle='round,pad=0.2', fc='#EDE9FE', ec=C_MAS, lw=1))

# arrow: selection → MAS
arrow(ax, 6.65, 6.15, 6.9, 6.5, 'team', C_SEL, lw=2)

# ══════════════════════════════════════════════════════════════════════════
# 5. CO-DREAM  (bottom-center)
# ══════════════════════════════════════════════════════════════════════════
dr_bg = FancyBboxPatch((4.0, 0.4), 6.8, 3.3,
                       boxstyle="round,pad=0.1,rounding_size=0.4",
                       linewidth=2.2, edgecolor=C_DREAM, facecolor=C_BGDR, zorder=2)
ax.add_patch(dr_bg)
ax.text(7.4, 3.4, 'Co-Dream  (offline joint imagination session)',
        ha='center', va='center', fontsize=10, fontweight='bold', color=C_DREAM)

phases = [
    ('① REFLECT', 'What surprised each agent?', '#1D4ED8'),
    ('② CONTRAST', 'Compare to best performer\n(MemCollab-inspired, stays private)', '#0891B2'),
    ('③ IMAGINE', 'Bold "what if" hypotheses\ngrounded in contrast deltas', '#059669'),
    ('④ DEBATE', 'Asymmetric: A challenges B\nonly in B\'s strong domains', '#7C3AED'),
    ('⑤ CRYSTALLIZE', 'Private novel insight →\nskill updates + hypotheses', '#B45309'),
]
ph_x = [4.3, 5.58, 6.86, 8.14, 9.42]
for i, ((ph, sub, pc), px) in enumerate(zip(phases, ph_x)):
    box(ax, px-0.56, 0.6, 1.1, 2.55, pc, 'white', ph, sub, radius=0.22, lw=1.6)
    if i < 4:
        arrow(ax, px+0.54, 1.87, ph_x[i+1]-0.56, 1.87, '', pc, lw=1.6)

# arrow: MAS → Co-Dream
arrow(ax, 8.95, 4.0, 7.5, 3.7, 'task done', C_MAS, lw=2, curve=0.2)

# diverging arrows back to pool (private)
for i, (px, py) in enumerate([(4.8, 0.6), (6.1, 0.6), (7.4, 0.6)]):
    ax.annotate('', xy=(1.95, 1.9+i*0.35), xytext=(px, py),
                arrowprops=dict(arrowstyle='->', color=C_DREAM, lw=1.4,
                                mutation_scale=14,
                                connectionstyle='arc3,rad=0.35'), zorder=5)
ax.text(3.7, 1.05, '← private\n    updates', ha='center', va='center',
        fontsize=7, color=C_DREAM, style='italic')

# ══════════════════════════════════════════════════════════════════════════
# 6. POOL LIFECYCLE  (right)
# ══════════════════════════════════════════════════════════════════════════
lc_bg = FancyBboxPatch((11.3, 1.2), 6.3, 7.8,
                       boxstyle="round,pad=0.1,rounding_size=0.4",
                       linewidth=2, edgecolor=C_LC, facecolor=C_BGLC, zorder=1)
ax.add_patch(lc_bg)
ax.text(14.45, 8.7, 'Pool Lifecycle Operators', ha='center', va='center',
        fontsize=10, fontweight='bold', color=C_LC)
ax.text(14.45, 8.3, '(triggered every 10 tasks via check_and_apply_lifecycle)',
        ha='center', va='center', fontsize=7, color='#92400E')

ops = [
    ('FORK',    'Divergent specialist\nfrom high-entropy agent',  '🌿', 7.3),
    ('MERGE',   'Similar agents → one\nconsolidated specialist',  '🔗', 5.7),
    ('PRUNE',   'Remove persistently\nunderperforming agent',     '✂️', 4.1),
    ('GENESIS', 'Spawn new agent for\nunder-served task type',    '✨', 2.5),
]
for op, desc, icon, oy in ops:
    box(ax, 11.7, oy-0.55, 5.4, 1.05, C_LC, 'white', f'{icon}  {op}', desc, radius=0.28, lw=1.8)

# tiny before/after diagrams for FORK
def small_circle(cx, cy, r=0.18, fc='white', ec=C_LC, lw=1.2):
    ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=6))

# FORK illustration (right of box)
small_circle(16.6, 7.5)
ax.text(16.6, 7.5, 'A', ha='center', va='center', fontsize=6, color=C_LC)
arrow(ax, 16.8, 7.5, 17.0, 7.75, '', C_LC, lw=1.2)
arrow(ax, 16.8, 7.5, 17.0, 7.25, '', C_LC, lw=1.2)
small_circle(17.15, 7.75, fc='#DBEAFE'); ax.text(17.15, 7.75, 'A1', ha='center', va='center', fontsize=5.5, color=C_LC)
small_circle(17.15, 7.25, fc='#DBEAFE'); ax.text(17.15, 7.25, 'A2', ha='center', va='center', fontsize=5.5, color=C_LC)

# MERGE illustration
small_circle(16.55, 5.95)
small_circle(16.55, 5.45)
ax.text(16.55, 5.95, 'A', ha='center', va='center', fontsize=6, color=C_LC)
ax.text(16.55, 5.45, 'B', ha='center', va='center', fontsize=6, color=C_LC)
arrow(ax, 16.73, 5.95, 16.93, 5.7, '', C_LC, lw=1.2)
arrow(ax, 16.73, 5.45, 16.93, 5.7, '', C_LC, lw=1.2)
small_circle(17.1, 5.7, fc='#FED7AA'); ax.text(17.1, 5.7, 'AB', ha='center', va='center', fontsize=5.5, color=C_LC)

# arrow: MAS result → lifecycle
arrow(ax, 11.0, 6.5, 11.3, 6.5, 'pool\nupdate', C_LC, lw=2)

# ══════════════════════════════════════════════════════════════════════════
# 7. LEGEND / FOOTER
# ══════════════════════════════════════════════════════════════════════════
legend_items = [
    (C_POOL,  'Agent Pool'),
    (C_SEL,   'Team Selection'),
    (C_MAS,   'Leader MAS'),
    (C_DREAM, 'Co-Dream'),
    (C_LC,    'Lifecycle'),
]
for i, (lc, lt) in enumerate(legend_items):
    lx = 1.0 + i*3.1
    ax.add_patch(plt.Rectangle((lx, 0.05), 0.35, 0.2, facecolor=lc, zorder=6))
    ax.text(lx+0.45, 0.15, lt, va='center', fontsize=8, color=C_TEXT)

plt.tight_layout(pad=0.3)
plt.savefig('figures/evopool_architecture.png', dpi=180, bbox_inches='tight',
            facecolor='white')
print("Saved: figures/evopool_architecture.png")
