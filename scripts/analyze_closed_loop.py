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

# Helper for consistent colors
def color_map(types):
    return ['#2ecc71' if t == 'pytorch' else '#3498db' for t in types]

# 1. Success Rate (higher is better)
top_success = df.nlargest(5, 'Success Rate')
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(top_success)), top_success['Success Rate'], color=color_map(top_success['Type']))
ax.set_xticks(range(len(top_success)))
ax.set_xticklabels(top_success['Model'], rotation=45, ha='right')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Top 5 - Success Rate', fontweight='bold')
ax.set_ylim([0, 1])
for i, (idx, row) in enumerate(top_success.iterrows()):
    ax.text(i, row['Success Rate'] + 0.02, f"{row['Success Rate']:.1f}%", ha='center', va='bottom')
ax.grid(axis='y', alpha=0.3)
fig.savefig('results/closed_loop_success.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to results/closed_loop_success.png")

# 2. Final Error (lower is better)
top_error = df.nsmallest(5, 'Final Error (m)')
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(top_error)), top_error['Final Error (m)'], color=color_map(top_error['Type']))
ax.set_xticks(range(len(top_error)))
ax.set_xticklabels(top_error['Model'], rotation=45, ha='right')
ax.set_ylabel('Mean Final EE Error (m)')
ax.set_title('Top 5 - Final Error (Lower Better)', fontweight='bold')
ax.set_ylim([0, top_error['Final Error (m)'].max() + 0.1])
for i, (idx, row) in enumerate(top_error.iterrows()):
    ax.text(i, row['Final Error (m)'] + 0.02, f"{row['Final Error (m)']:.4f}m", ha='center', va='bottom')
ax.grid(axis='y', alpha=0.3)
fig.savefig('results/closed_loop_final_error.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to results/closed_loop_final_error.png")

# 3. Solve Time (lower is better)
top_time = df.nsmallest(5, 'Solve Time (ms)')
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(top_time)), top_time['Solve Time (ms)'], color=color_map(top_time['Type']))
ax.set_xticks(range(len(top_time)))
ax.set_xticklabels(top_time['Model'], rotation=45, ha='right')
ax.set_ylabel('Mean Solve Time (ms)')
ax.set_ylim([0, top_time['Solve Time (ms)'].max() + 1])
ax.set_title('Top 5 - Computational Efficiency (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_time.iterrows()):
    ax.text(i, row['Solve Time (ms)'] + 0.05, f"{row['Solve Time (ms)']:.3f}ms", ha='center', va='bottom')
ax.grid(axis='y', alpha=0.3)
fig.savefig('results/closed_loop_solvetime.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to results/closed_loop_solvetime.png")

# 4. CPU Percent (lower is better)
top_cpu = df.nsmallest(5, 'CPU Percent')
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(top_cpu)), top_cpu['CPU Percent'], color=color_map(top_cpu['Type']))
ax.set_xticks(range(len(top_cpu)))
ax.set_xticklabels(top_cpu['Model'], rotation=45, ha='right')
ax.set_ylabel('CPU Utilization (%)')
ax.set_ylim([0, top_cpu['CPU Percent'].max() + 1])
ax.set_title('Top 5 - CPU Efficiency (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_cpu.iterrows()):
    ax.text(i, row['CPU Percent'] + 0.1, f"{row['CPU Percent']:.2f}%", ha='center', va='bottom')
ax.grid(axis='y', alpha=0.3)
fig.savefig('results/closed_loop_cpu.png', dpi=150, bbox_inches='tight')
print("✓ Plot saved to results/closed_loop_cpu.png")

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Top 5 Models by Metric")
print("="*80)

print("\nSUCCESS RATE (Higher is Better)")
print(top_success[['Model', 'Type', 'Success Rate']].to_string(index=False))

print("\nFINAL ERROR (Lower is Better)")
print(top_error[['Model', 'Type', 'Final Error (m)']].to_string(index=False))

print("\nSOLVE TIME (Lower is Better)")
print(top_time[['Model', 'Type', 'Solve Time (ms)']].to_string(index=False))

print("\nCPU UTILIZATION (Lower is Better)")
print(top_cpu[['Model', 'Type', 'CPU Percent']].to_string(index=False))
