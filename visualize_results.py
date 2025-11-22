#!/usr/bin/env python3
"""
Performance Visualization for ApSpGEMM GPU Implementation
Creates publication-quality plots similar to ACM paper figures
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams

# Set publication-quality defaults
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13

# Load benchmark data
df = pd.read_csv('benchmark_results.csv')

# Calculate compression ratio (Output_NNZ / Input_NNZ)
df['Compression_Ratio'] = df['Output_NNZ'] / df['Input_NNZ']

# Create figure with multiple subplots (2x2 grid)
fig = plt.figure(figsize=(14, 10))

# ============================================================================
# Plot 1: Speedup vs Matrix Size (Similar to paper's scaling analysis)
# ============================================================================
ax1 = plt.subplot(2, 2, 1)

matrix_sizes = df['Rows'].values
speedups = df['Speedup'].values
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(matrix_sizes)))

bars = ax1.bar(range(len(df)), speedups, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for i, (bar, speedup) in enumerate(zip(bars, speedups)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}×',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='CPU Baseline (1×)')
ax1.set_xlabel('Matrix Size (rows)', fontweight='bold')
ax1.set_ylabel('Speedup (GPU vs CPU)', fontweight='bold')
ax1.set_title('(a) GPU Speedup vs Matrix Size', fontweight='bold', loc='left')
ax1.set_xticks(range(len(df)))
ax1.set_xticklabels([f"{size}×{size}" for size in matrix_sizes], rotation=0)
ax1.grid(axis='y', alpha=0.3, linestyle=':')
ax1.legend(loc='upper left')
ax1.set_ylim(0, max(speedups) * 1.2)

# ============================================================================
# Plot 2: GFlops Comparison (CPU vs GPU) - Similar to paper's Figure 11(a)
# ============================================================================
ax2 = plt.subplot(2, 2, 2)

x = np.arange(len(df))
width = 0.35

bars1 = ax2.bar(x - width/2, df['GFlops_CPU'], width, label='CPU Baseline',
                color='#ff6b6b', edgecolor='black', linewidth=1)
bars2 = ax2.bar(x + width/2, df['GFlops_GPU'], width, label='GPU (Our Method)',
                color='#4ecdc4', edgecolor='black', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.005:  # Only show if meaningful
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}',
                     ha='center', va='bottom', fontsize=8)

ax2.set_xlabel('Matrix', fontweight='bold')
ax2.set_ylabel('Performance (GFlops)', fontweight='bold')
ax2.set_title('(b) Computational Performance (GFlops)', fontweight='bold', loc='left')
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace('synthetic_', '') for m in df['Matrix']], rotation=0)
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle=':')

# ============================================================================
# Plot 3: Execution Time Breakdown - Similar to paper's Figure 11(b)
# ============================================================================
ax3 = plt.subplot(2, 2, 3)

# For this visualization, we'll show CPU vs GPU time breakdown
cpu_times = df['CPU_ms'].values
gpu_times = df['GPU_ms'].values
total_times = cpu_times + gpu_times

# Calculate percentages
cpu_percentages = (cpu_times / (cpu_times + gpu_times)) * 100
gpu_percentages = (gpu_times / (cpu_times + gpu_times)) * 100

x = np.arange(len(df))
width = 0.6

# Stacked bar chart
p1 = ax3.bar(x, cpu_percentages, width, label='CPU Baseline Time',
             color='#ff6b6b', edgecolor='black', linewidth=1)
p2 = ax3.bar(x, gpu_percentages, width, bottom=cpu_percentages,
             label='GPU Kernel Time', color='#4ecdc4', edgecolor='black', linewidth=1)

# Add percentage labels
for i, (cpu_pct, gpu_pct) in enumerate(zip(cpu_percentages, gpu_percentages)):
    # CPU label
    ax3.text(i, cpu_pct/2, f'{cpu_pct:.1f}%',
             ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    # GPU label
    ax3.text(i, cpu_pct + gpu_pct/2, f'{gpu_pct:.1f}%',
             ha='center', va='center', fontsize=8, fontweight='bold', color='white')

ax3.set_xlabel('Matrix', fontweight='bold')
ax3.set_ylabel('Percentage of Total Time (%)', fontweight='bold')
ax3.set_title('(c) Execution Time Distribution', fontweight='bold', loc='left')
ax3.set_xticks(x)
ax3.set_xticklabels([m.replace('synthetic_', '') for m in df['Matrix']], rotation=0)
ax3.set_ylim(0, 100)
ax3.legend(loc='upper right')
ax3.grid(axis='y', alpha=0.3, linestyle=':')

# ============================================================================
# Plot 4: Performance vs Compression Ratio (Scatter plot like paper's main figure)
# ============================================================================
ax4 = plt.subplot(2, 2, 4)

compression = df['Compression_Ratio'].values
gflops_gpu = df['GFlops_GPU'].values
matrix_sizes = df['Rows'].values

# Create scatter plot with size proportional to matrix size
sizes = (matrix_sizes / matrix_sizes.min()) * 100
colors_scatter = plt.cm.plasma(np.linspace(0.2, 0.9, len(df)))

scatter = ax4.scatter(compression, gflops_gpu, s=sizes, c=colors_scatter,
                     alpha=0.7, edgecolors='black', linewidth=1.5)

# Add labels for each point
for i, (comp, gflop, matrix) in enumerate(zip(compression, gflops_gpu, df['Matrix'])):
    label = matrix.replace('synthetic_', '')
    ax4.annotate(label, (comp, gflop),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors_scatter[i], alpha=0.3))

# Fit a trend line
if len(compression) > 1:
    z = np.polyfit(compression, gflops_gpu, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(compression.min(), compression.max(), 100)
    ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')

ax4.set_xlabel('Compression Ratio (Output NNZ / Input NNZ)', fontweight='bold')
ax4.set_ylabel('GPU Performance (GFlops)', fontweight='bold')
ax4.set_title('(d) Performance vs Compression Ratio', fontweight='bold', loc='left')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3, linestyle=':')

# Add overall title
fig.suptitle('ApSpGEMM: GPU SpGEMM Performance Analysis\n' +
             'Gustavson\'s Algorithm - CUDA Implementation',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])

# Save figure
plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('performance_analysis.pdf', bbox_inches='tight')
print("✓ Saved: performance_analysis.png (300 DPI)")
print("✓ Saved: performance_analysis.pdf (vector format)")

# ============================================================================
# Additional Figure: Detailed Scaling Analysis
# ============================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Execution time comparison
matrix_labels = [m.replace('synthetic_', '') for m in df['Matrix']]
x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['CPU_ms'], width, label='CPU Time',
                color='#e74c3c', edgecolor='black', linewidth=1)
bars2 = ax1.bar(x + width/2, df['GPU_ms'], width, label='GPU Time',
                color='#3498db', edgecolor='black', linewidth=1)

ax1.set_xlabel('Matrix Size', fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontweight='bold')
ax1.set_title('(a) Execution Time: CPU vs GPU', fontweight='bold', loc='left')
ax1.set_xticks(x)
ax1.set_xticklabels(matrix_labels, rotation=0)
ax1.legend()
ax1.grid(axis='y', alpha=0.3, linestyle=':')
ax1.set_yscale('log')

# Right: Speedup trend with fitted curve
matrix_sizes_numeric = df['Rows'].values
speedups = df['Speedup'].values

# Remove outliers (speedup < 0.5) for trend fitting
valid_indices = speedups > 0.5
sizes_valid = matrix_sizes_numeric[valid_indices]
speedups_valid = speedups[valid_indices]

ax2.plot(matrix_sizes_numeric, speedups, 'o-', markersize=10, linewidth=2,
         color='#2ecc71', markeredgecolor='black', markeredgewidth=1.5,
         label='Measured Speedup')

# Fit polynomial trend (excluding outliers)
if len(sizes_valid) > 1:
    z = np.polyfit(np.log(sizes_valid), speedups_valid, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(matrix_sizes_numeric.min(), matrix_sizes_numeric.max(), 100)
    y_trend = p(np.log(x_trend))
    ax2.plot(x_trend, y_trend, '--', linewidth=2, color='red', alpha=0.7,
             label=f'Trend: {z[0]:.2f}×log(n) + {z[1]:.2f}')

# Add labels
for size, speedup in zip(matrix_sizes_numeric, speedups):
    ax2.annotate(f'{speedup:.2f}×', (size, speedup),
                xytext=(0, 10), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold')

ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No Speedup')
ax2.set_xlabel('Matrix Size (rows = cols)', fontweight='bold')
ax2.set_ylabel('Speedup (×)', fontweight='bold')
ax2.set_title('(b) Scaling Analysis: Speedup vs Matrix Size', fontweight='bold', loc='left')
ax2.set_xscale('log')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3, linestyle=':')

fig2.suptitle('Detailed Performance Scaling Analysis',
              fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig('scaling_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('scaling_analysis.pdf', bbox_inches='tight')
print("✓ Saved: scaling_analysis.png (300 DPI)")
print("✓ Saved: scaling_analysis.pdf (vector format)")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("PERFORMANCE SUMMARY STATISTICS")
print("="*70)
print(f"Average Speedup: {df['Speedup'].mean():.2f}×")
print(f"Maximum Speedup: {df['Speedup'].max():.2f}× (Matrix: {df.loc[df['Speedup'].idxmax(), 'Matrix']})")
print(f"Average GPU GFlops: {df['GFlops_GPU'].mean():.3f}")
print(f"Peak GPU GFlops: {df['GFlops_GPU'].max():.3f} (Matrix: {df.loc[df['GFlops_GPU'].idxmax(), 'Matrix']})")
print(f"Average Compression Ratio: {df['Compression_Ratio'].mean():.2f}")
print(f"Total Matrices Tested: {len(df)}")
print("="*70)

plt.show()
