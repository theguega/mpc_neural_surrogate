"""
Analyze and compare closed-loop evaluation results.
Plots top 3 models for success rate, final error, solve time, and CPU utilization.
"""

import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load all JSON files from results/closed_loop/
results_dir = 'results/closed_loop'
json_files = glob.glob(os.path.join(results_dir, '*.json'))

print(f"Found {len(json_files)} evaluation results")

# Parse results
data = []
for filepath in json_files:
    with open(filepath, 'r') as f:
        result = json.load(f)
    
    # Extract model info and aggregate metrics
    model_path = result.get('model_path', 'MPC')
    model_name = os.path.splitext(os.path.basename(model_path))[0] if model_path else 'MPC'
    controller_type = result.get('controller_type', 'unknown')
    
    agg = result.get('aggregate_metrics', {})
    
    data.append({
        'Model': model_name,
        'Type': controller_type,
        'Success Rate': agg.get('success_rate', 0),
        'Final Error (m)': agg.get('mean_final_ee_error', 0),
        'Solve Time (ms)': agg.get('mean_solve_time', 0) * 1000,  # Convert to ms
        'CPU Percent': agg.get('mean_cpu_percent', 0)
    })

df = pd.DataFrame(data)
print(f"\nLoaded {len(df)} models")
print(df)

# Get top 3 for each metric
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Top 3 Models by Key Metrics', fontsize=16, fontweight='bold')

# 1. Success Rate (higher is better)
ax = axes[0, 0]
top_success = df.nlargest(3, 'Success Rate')
colors = ['#2ecc71' if t == 'pytorch' else '#3498db' for t in top_success['Type']]
bars = ax.barh(range(len(top_success)), top_success['Success Rate'], color=colors)
ax.set_yticks(range(len(top_success)))
ax.set_yticklabels(top_success['Model'])
ax.set_xlabel('Success Rate (%)')
ax.set_title('Top 3 - Success Rate', fontweight='bold')
ax.set_xlim([0, 100])
for i, (idx, row) in enumerate(top_success.iterrows()):
    ax.text(row['Success Rate'] + 2, i, f"{row['Success Rate']:.1f}%", va='center')
ax.grid(axis='x', alpha=0.3)

# 2. Final Error (lower is better)
ax = axes[0, 1]
top_error = df.nsmallest(3, 'Final Error (m)')
colors = ['#2ecc71' if t == 'pytorch' else '#3498db' for t in top_error['Type']]
bars = ax.barh(range(len(top_error)), top_error['Final Error (m)'], color=colors)
ax.set_yticks(range(len(top_error)))
ax.set_yticklabels(top_error['Model'])
ax.set_xlabel('Mean Final EE Error (m)')
ax.set_title('Top 3 - Final Error (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_error.iterrows()):
    ax.text(row['Final Error (m)'] + 0.02, i, f"{row['Final Error (m)']:.4f}m", va='center')
ax.grid(axis='x', alpha=0.3)

# 3. Solve Time (lower is better)
ax = axes[1, 0]
top_time = df.nsmallest(3, 'Solve Time (ms)')
colors = ['#2ecc71' if t == 'pytorch' else '#3498db' for t in top_time['Type']]
bars = ax.barh(range(len(top_time)), top_time['Solve Time (ms)'], color=colors)
ax.set_yticks(range(len(top_time)))
ax.set_yticklabels(top_time['Model'])
ax.set_xlabel('Mean Solve Time (ms)')
ax.set_title('Top 3 - Computational Efficiency (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_time.iterrows()):
    ax.text(row['Solve Time (ms)'] + 0.05, i, f"{row['Solve Time (ms)']:.3f}ms", va='center')
ax.grid(axis='x', alpha=0.3)

# 4. CPU Percent (lower is better)
ax = axes[1, 1]
top_cpu = df.nsmallest(3, 'CPU Percent')
colors = ['#2ecc71' if t == 'pytorch' else '#3498db' for t in top_cpu['Type']]
bars = ax.barh(range(len(top_cpu)), top_cpu['CPU Percent'], color=colors)
ax.set_yticks(range(len(top_cpu)))
ax.set_yticklabels(top_cpu['Model'])
ax.set_xlabel('CPU Utilization (%)')
ax.set_title('Top 3 - CPU Efficiency (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_cpu.iterrows()):
    ax.text(row['CPU Percent'] + 0.1, i, f"{row['CPU Percent']:.2f}%", va='center')
ax.grid(axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='PyTorch'),
    Patch(facecolor='#3498db', label='Scikit-Learn')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2)

plt.tight_layout()
plt.savefig('results/closed_loop_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Plot saved to results/closed_loop_comparison.png")
plt.show()

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Top 3 Models by Metric")
print("="*80)

print("\nSUCCESS RATE (Higher is Better)")
print(top_success[['Model', 'Type', 'Success Rate']].to_string(index=False))

print("\nFINAL ERROR (Lower is Better)")
print(top_error[['Model', 'Type', 'Final Error (m)']].to_string(index=False))

print("\nSOLVE TIME (Lower is Better)")
print(top_time[['Model', 'Type', 'Solve Time (ms)']].to_string(index=False))

print("\nCPU UTILIZATION (Lower is Better)")
print(top_cpu[['Model', 'Type', 'CPU Percent']].to_string(index=False))
