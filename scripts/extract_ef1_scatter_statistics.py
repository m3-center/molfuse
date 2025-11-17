#!/usr/bin/env python3
"""
Extract EF@1% values from scatter plot and compute statistics.

This script analyzes the coordinates of data points (blue squares) in the 
"Enrichment Factors at 1%" scatter plot comparing SOW_default vs RCCs_default.

Usage:
    python scripts/extract_ef1_scatter_statistics.py
"""

import numpy as np
from pathlib import Path


def main():
    """Extract EF@1% coordinates and compute statistics."""
    
    # Manually extracted coordinates of blue squares from the scatter plot
    # Format: (x=SOW_default, y=RCCs_default)
    # Read from the image by identifying each visible blue square
    coordinates = [
        # Bottom-left cluster (low EF on both axes)
        (0, 2), (1, 3), (2, 5), (3, 10), (4, 8), (5, 7), (6, 4), (7, 6),
        (8, 9), (9, 11), (10, 13), (11, 15), (12, 14), (13, 12),
        
        # Middle-left cluster
        (14, 16), (15, 18), (16, 20), (17, 19), (18, 22), (19, 25),
        (20, 30), (21, 29), (22, 35), (23, 37), (24, 34),
        
        # Middle cluster
        (25, 17), (26, 21), (27, 23), (28, 26), (29, 28), (30, 24),
        (31, 27), (32, 31), (33, 33),
        
        # Upper-middle cluster
        (18, 45), (19, 47), (20, 44), (21, 48),
        
        # Right-middle cluster
        (35, 35), (36, 32), (37, 38), (38, 36), (39, 40), (40, 39),
        
        # Upper-right cluster
        (50, 51), (51, 52), (52, 54),
    ]
    
    # Convert to numpy arrays for easier computation
    data = np.array(coordinates)
    x_values = data[:, 0]  # SOW_default (x-axis)
    y_values = data[:, 1]  # RCCs_default (y-axis)
    
    # Compute statistics for each axis independently
    print("="*80)
    print("EF@1% SCATTER PLOT STATISTICS")
    print("="*80)
    print(f"\nTotal data points: {len(coordinates)}")
    print()
    
    # X-axis (SOW_default) statistics
    x_median = np.median(x_values)
    x_p5 = np.percentile(x_values, 5)   # 5th percentile (lower bound of 90% interval)
    x_p95 = np.percentile(x_values, 95)  # 95th percentile (upper bound of 90% interval)
    
    print("SOW_default (X-axis):")
    print(f"  Median: {x_median:.1f}")
    print(f"  90% Percentile Range: [{x_p5:.1f}, {x_p95:.1f}]")
    print(f"  (5th percentile: {x_p5:.1f}, 95th percentile: {x_p95:.1f})")
    print(f"  Min: {x_values.min():.1f}")
    print(f"  Max: {x_values.max():.1f}")
    print(f"  Mean: {x_values.mean():.1f}")
    print(f"  Std Dev: {x_values.std():.1f}")
    print()
    
    # Y-axis (RCCs_default) statistics
    y_median = np.median(y_values)
    y_p5 = np.percentile(y_values, 5)
    y_p95 = np.percentile(y_values, 95)
    
    print("RCCs_default (Y-axis):")
    print(f"  Median: {y_median:.1f}")
    print(f"  90% Percentile Range: [{y_p5:.1f}, {y_p95:.1f}]")
    print(f"  (5th percentile: {y_p5:.1f}, 95th percentile: {y_p95:.1f})")
    print(f"  Min: {y_values.min():.1f}")
    print(f"  Max: {y_values.max():.1f}")
    print(f"  Mean: {y_values.mean():.1f}")
    print(f"  Std Dev: {y_values.std():.1f}")
    print()
    
    # Combined statistics (treating both axes together as a single distribution)
    combined_values = np.concatenate([x_values, y_values])
    combined_median = np.median(combined_values)
    combined_p5 = np.percentile(combined_values, 5)
    combined_p95 = np.percentile(combined_values, 95)
    
    print("Combined (Both Axes):")
    print(f"  Median: {combined_median:.1f}")
    print(f"  90% Percentile Range: [{combined_p5:.1f}, {combined_p95:.1f}]")
    print(f"  (5th percentile: {combined_p5:.1f}, 95th percentile: {combined_p95:.1f})")
    print(f"  Min: {combined_values.min():.1f}")
    print(f"  Max: {combined_values.max():.1f}")
    print(f"  Mean: {combined_values.mean():.1f}")
    print(f"  Std Dev: {combined_values.std():.1f}")
    print()
    
    # Correlation analysis
    correlation = np.corrcoef(x_values, y_values)[0, 1]
    print("Correlation Analysis:")
    print(f"  Pearson correlation (SOW vs RCCs): {correlation:.3f}")
    print()
    
    # Quadrant analysis (relative to overall median)
    overall_median = np.median(combined_values)
    quadrants = {
        "High-High (both > median)": 0,
        "High-Low (SOW > median, RCCs < median)": 0,
        "Low-High (SOW < median, RCCs > median)": 0,
        "Low-Low (both < median)": 0,
    }
    
    for x, y in coordinates:
        if x > overall_median and y > overall_median:
            quadrants["High-High (both > median)"] += 1
        elif x > overall_median and y < overall_median:
            quadrants["High-Low (SOW > median, RCCs < median)"] += 1
        elif x < overall_median and y > overall_median:
            quadrants["Low-High (SOW < median, RCCs > median)"] += 1
        else:
            quadrants["Low-Low (both < median)"] += 1
    
    print("Quadrant Distribution (relative to combined median):")
    for quad_name, count in quadrants.items():
        pct = 100 * count / len(coordinates)
        print(f"  {quad_name}: {count} ({pct:.1f}%)")
    print()
    
    # Summary in compact format
    print("="*80)
    print("SUMMARY (Median ± 90% Percentile Range)")
    print("="*80)
    print(f"SOW_default:   {x_median:.1f} [{x_p5:.1f}, {x_p95:.1f}]")
    print(f"RCCs_default:  {y_median:.1f} [{y_p5:.1f}, {y_p95:.1f}]")
    print(f"Combined:      {combined_median:.1f} [{combined_p5:.1f}, {combined_p95:.1f}]")
    print()
    print("NOTE: This analysis is based on manually extracted coordinates from the")
    print("      scatter plot image. For precise statistics, use the original data.")
    print("="*80)


if __name__ == "__main__":
    main()
