import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D  # Not needed for 2D
# from matplotlib.lines import Line2D      # Not needed for 2D

# Constants
D = 2000      # Original Features
d = 2         # UMAP Dimensions

# Ranges
# N: 1k to 1M
# M: 1 to 10k (Extended range to see crossover)
N_values = np.geomspace(10, 10000000, num=50)
M_values = np.geomspace(10, 10000000, num=50)

N, M = np.meshgrid(N_values, M_values)

# 1. Direct Measurement: O(M * N * D)
ops_direct = M * N * D

# 2. UMAP Training Phase: Empirical O(N^1.14 * D)
# Removed overhead K as requested
ops_umap_train = (np.power(N, 1.14) * D)

# 3. UMAP Inference (Transform + Search): 
# Scenario A: Ideal Low-Dim / Optimized Index (e.g., HNSW, Fingerprints) -> O(log N)
# ops_umap_inference_ideal = (M * np.log2(N) * D) + (M * np.log2(N) * d)

# Scenario B: Realistic High-Dim Features (Curse of Dimensionality)
# In D=2000, approximate NN search degrades to linear scan O(N).
# We also add a constant overhead 'K_graph' for the graph traversal overhead vs BLAS.
K_graph = 2.0 # Graph traversal is slower than pure matrix mult
ops_umap_inference = (M * K_graph * N * D) + (M * np.log2(N) * d)

# Note on Direct Search with Trees:
# In high dimensions (D=2000), KDTree/BallTree performance degrades to O(N) 
# due to the curse of dimensionality. Thus, Direct remains O(M * N * D).

# 4. UMAP Total (Training + Inference)
ops_umap_total = ops_umap_train + ops_umap_inference

# 5. Calculate Speedup (Direct / UMAP)
speedup = ops_direct / ops_umap_total
log_speedup = np.log10(speedup)

# Plotting: 2D Phase Diagram (Heatmap)
plt.figure(figsize=(10, 8))

# Use Log10 coordinates for plotting to ensure smooth contours
X_log = np.log10(N)
Y_log = np.log10(M)

# Create Heatmap
max_abs_log = np.max(np.abs(log_speedup))
limit = max(2, max_abs_log) 

plt.pcolormesh(X_log, Y_log, log_speedup, cmap='RdBu', shading='auto', 
               vmin=-limit, vmax=limit)

# Add Colorbar
cbar = plt.colorbar()
cbar.set_label('Log10(Speedup Factor) [Blue=UMAP Faster]', fontsize=12)
cbar_ticks = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'$10^{{{t}}}$x' for t in cbar_ticks])
cbar.ax.set_ylim(-3, limit)

# Add Contour Line for Breakeven (Speedup = 1 => log_speedup = 0)
CS = plt.contour(X_log, Y_log, log_speedup, levels=[0], colors='black', linewidths=3, linestyles='-')
plt.clabel(CS, inline=True, fontsize=12, fmt={0: 'Breakeven (1x)'})

# Add Contour Lines for 10x, 100x, and 1000x speedup
CS2 = plt.contour(X_log, Y_log, log_speedup, levels=[1, 2, 3, 4, 5], colors='black', linewidths=2, linestyles='--')
plt.clabel(CS2, inline=True, fontsize=10, fmt={1: '10x', 2: '100x', 3: '1,000x', 4: '10,000x', 5: '100,000x'})

# Axes and Labels
# Manually set ticks to simulate log scale
x_ticks = np.arange(int(np.min(X_log)), int(np.max(X_log)) + 1)
y_ticks = np.arange(int(np.min(Y_log)), int(np.max(Y_log)) + 1)

plt.xticks(x_ticks, [f'$10^{{{t}}}$' for t in x_ticks])
plt.yticks(y_ticks, [f'$10^{{{t}}}$' for t in y_ticks])

plt.xlabel('Number of Samples in RS + ZINC (N)', fontsize=12)
plt.ylabel('Number of Candidates (M)', fontsize=12)
plt.title('Phase Diagram: Direct Search vs. UMAP\n(Realistic High-Dim: Transform is O(N))', fontsize=14)


# Annotations for Regimes
# Position relative to the log axes
x_min, x_max = np.min(X_log), np.max(X_log)
y_min, y_max = np.min(Y_log), np.max(Y_log)

#plt.text(x_min + 0.5, y_min + 0.5, 'Direct is Faster\n(Low M)', 
#         color='darkred', fontweight='bold', ha='left', va='bottom', 
#         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

#plt.text(x_max - 0.5, y_max - 0.5, 'UMAP is Faster\n(High M)', 
#         color='darkblue', fontweight='bold', ha='right', va='top',
#         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

plt.tight_layout()

# Save plot
plt.savefig('benchmark_theory_phase_diagram.png', dpi=300)
plt.savefig('benchmark_theory_phase_diagram.pdf')

plt.show()

# Generate LaTeX description
latex_content = r"""
\documentclass{article}
\usepackage{amsmath}
\begin{document}
\section*{Complexity Comparison: Direct Search vs. UMAP}

\subsection*{Definitions}
\begin{itemize}
    \item $N$: Number of samples in the Reference Set + ZINC (Training Set).
    \item $M$: Number of candidates (Actives) to screen.
    \item $D$: Original dimensionality of the feature space (e.g., 2000).
    \item $d$: Reduced dimensionality (e.g., 2).
\end{itemize}

\subsection*{1. Direct Measurement (Baseline)}
The complexity of performing exact 1-Nearest Neighbor search in the original high-dimensional space is linear with respect to all variables:
\[ O(M \cdot N \cdot D) \]

\subsection*{2. UMAP Approach}
The UMAP-based screening pipeline consists of two distinct phases:

\subsubsection*{Training Phase (Offline)}
The model is trained on the combined Reference Set and ZINC decoys. The empirical complexity for UMAP training is super-linear in $N$ but independent of $M$:
\[ O(N^{1.14} \cdot D) \]

\subsubsection*{Inference Phase (Online)}
The inference phase involves projecting the $M$ candidates into the low-dimensional space and performing the search.
\begin{itemize}
    \item \textbf{Transform:} Projecting points using approximate nearest neighbors: $O(M \cdot \log(N) \cdot D)$
    \item \textbf{Search:} Performing 1-NN search in the reduced 2D space using a KDTree/BallTree: $O(M \cdot \log(N) \cdot d)$
\end{itemize}
Total Inference Complexity:
\[ O(M \cdot \log(N) \cdot (D + d)) \]

\subsection*{3. Speedup Factor}
The theoretical speedup is defined as:
\[ \text{Speedup} = \frac{\text{Cost}_{\text{Direct}}}{\text{Cost}_{\text{UMAP Train}} + \text{Cost}_{\text{UMAP Inference}}} \]

For large $N$ and $M$, the UMAP approach provides significant acceleration because the search step complexity reduces from $O(N)$ to $O(\log N)$, and the high-dimensional distance calculations are amortized during the training phase.

\end{document}
"""

with open('benchmark_theory_complexity.tex', 'w') as f:
    f.write(latex_content)

print("LaTeX description saved to benchmark_theory_complexity.tex")