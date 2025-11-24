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
# Transform: M * log(N) * D (Projecting new points)
# Search in 2D: M * N * d (Exact search in low dim)
ops_umap_inference = (M * np.log2(N) * D) + (M * N * d)

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
cbar.set_label('Log10(Speedup Factor) [Red=Direct Faster, Blue=UMAP Faster]', fontsize=12)
cbar_ticks = [-2, -1, 0, 1, 2, 3]
cbar.set_ticks(cbar_ticks)
cbar.set_ticklabels([f'$10^{{{t}}}$x' for t in cbar_ticks])

# Add Contour Line for Breakeven (Speedup = 1 => log_speedup = 0)
CS = plt.contour(X_log, Y_log, log_speedup, levels=[0], colors='black', linewidths=3, linestyles='-')
plt.clabel(CS, inline=True, fontsize=12, fmt={0: 'Breakeven (1x)'})

# Add Contour Lines for 10x, 100x, and 1000x speedup
CS2 = plt.contour(X_log, Y_log, log_speedup, levels=[1, 2, 3], colors='black', linewidths=2, linestyles='--')
plt.clabel(CS2, inline=True, fontsize=10, fmt={1: '10x', 2: '100x', 3: '1000x'})

# Axes and Labels
# Manually set ticks to simulate log scale
x_ticks = np.arange(int(np.min(X_log)), int(np.max(X_log)) + 1)
y_ticks = np.arange(int(np.min(Y_log)), int(np.max(Y_log)) + 1)

plt.xticks(x_ticks, [f'$10^{{{t}}}$' for t in x_ticks])
plt.yticks(y_ticks, [f'$10^{{{t}}}$' for t in y_ticks])

plt.xlabel('Number of Samples (N)', fontsize=12)
plt.ylabel('Number of Candidates (M)', fontsize=12)
plt.title('Phase Diagram: Direct Search vs. UMAP\n(No Training Overhead)', fontsize=14)

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