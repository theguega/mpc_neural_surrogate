"""
Analyze and compare closed-loop evaluation results.
Plots top 3 models for success rate, final error, solve time, and CPU utilization.
"""

import json
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

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
        'Success Rate @0.02m': agg.get('success_rate', 0),
        'Success Rate @0.03m': agg.get('success_rate_relaxed', 0),
        'Success Rate @0.05m': agg.get('success_rate_loose', 0),
        'Final Error (m)': agg.get('mean_final_ee_error', 0),
        'Solve Time (ms)': agg.get('mean_solve_time', 0) * 1000,  # Convert to ms
        'CPU Time (s)': agg.get('mean_cpu_time', 0),
        'CPU Percent': agg.get('mean_cpu_percent', 0),
        'Steps to Success @0.02m': agg.get('mean_steps_to_success', 0),
        'Steps to Success @0.03m': agg.get('mean_steps_to_success_relaxed', 0),
        'Steps to Success @0.05m': agg.get('mean_steps_to_success_loose', 0),
        'Steps (All Episodes)': agg.get('mean_steps_all_episodes', 0),
        'Tracking Error (m)': agg.get('mean_tracking_error', 0),
        'Control Effort': agg.get('mean_control_effort', 0),
        'Mean Joint Error (MPC only)': agg.get('mean_joint_error', None),
        'Action Diff from MPC': agg.get('mean_action_diff_from_mpc', 0),
        'Wall Time (s)': agg.get('mean_wall_time', 0)
    })

df = pd.DataFrame(data)
print(f"\nLoaded {len(df)} runs")
print(df)


def canonical_model_name(name: str) -> str:
    """Collapse run-level model names into a base name for aggregation."""
    if name is None:
        return "Unknown"
    if name.lower() == "mpc":
        return "MPC"
    base = name
    base = re.sub(r"_best_[^_]+_\d+$", "", base)
    base = re.sub(r"_best_mse_\d+$", "", base)
    base = re.sub(r"_trial\d+.*$", "", base)
    base = re.sub(r"_\d{8}_\d{6}$", "", base)
    base = re.sub(r"_\d+$", "", base)
    return base


# Aggregate metrics across runs of the same model
df['ModelBase'] = df['Model'].apply(canonical_model_name)
metric_cols = [
    'Success Rate @0.02m', 'Success Rate @0.03m', 'Success Rate @0.05m',
    'Final Error (m)', 'Solve Time (ms)', 'CPU Time (s)', 'CPU Percent',
    'Steps to Success @0.02m', 'Steps to Success @0.03m', 'Steps to Success @0.05m',
    'Steps (All Episodes)', 'Tracking Error (m)', 'Control Effort',
    'Mean Joint Error (MPC only)', 'Action Diff from MPC', 'Wall Time (s)'
]

df_mean = df.groupby(['ModelBase', 'Type'])[metric_cols].mean().reset_index()
df_std = df.groupby(['ModelBase', 'Type'])[metric_cols].std().reset_index()

for col in metric_cols:
    df_mean[f"{col} Std"] = df_std[col]

df = df_mean.rename(columns={'ModelBase': 'Model'})
df.fillna(0, inplace=True)
print(f"Aggregated to {len(df)} models (mean over runs). Columns with 'Std' show run-to-run std dev.")
print(df)

# Helper for consistent colors and ensuring MPC inclusion
def color_map(types):
    palette = {
        'pytorch': '#2ecc71',
        'sklearn': '#3498db',
        'mpc': '#e67e22'
    }
    return [palette.get(t, '#7f8c8d') for t in types]

legend_elements = [
    Patch(facecolor='#2ecc71', label='PyTorch'),
    Patch(facecolor='#3498db', label='Scikit-Learn'),
    Patch(facecolor='#e67e22', label='MPC')
]

def select_top_with_mpc(df_metric: pd.DataFrame, metric: str, n: int = 5, ascending: bool = False):
    sorted_df = df_metric.sort_values(metric, ascending=ascending)
    if not ascending:
        sorted_df = df_metric.sort_values(metric, ascending=False)
    top = sorted_df.head(n).copy()
    mpc_rows = df_metric[df_metric['Type'] == 'mpc']
    if not mpc_rows.empty and not top['Model'].isin(mpc_rows['Model']).any():
        top = pd.concat([top, mpc_rows.head(1)], ignore_index=True)
    return top

# 1. Success Rate (all thresholds, higher is better)
top_success = select_top_with_mpc(df, 'Success Rate @0.02m', n=5, ascending=False).copy()
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(top_success))
width = 0.25

top_success['Success Rate @0.02m (%)'] = top_success['Success Rate @0.02m'] * 100
top_success['Success Rate @0.03m (%)'] = top_success['Success Rate @0.03m'] * 100
top_success['Success Rate @0.05m (%)'] = top_success['Success Rate @0.05m'] * 100

ax.bar(x_pos - width, top_success['Success Rate @0.02m (%)'], width, label='@0.02m (strict)', color='#e74c3c')
ax.bar(x_pos, top_success['Success Rate @0.03m (%)'], width, label='@0.03m (moderate)', color='#f39c12')
ax.bar(x_pos + width, top_success['Success Rate @0.05m (%)'], width, label='@0.05m (relaxed)', color='#27ae60')

ax.set_xticks(x_pos)
ax.set_xticklabels(top_success['Model'], rotation=45, ha='right')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Top 5 - Success Rate (All Thresholds)', fontweight='bold')
ax.set_ylim([0, 100])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('results/closed_loop/plots/mean/closed_loop_success_thresholds.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/closed_loop/plots/mean/closed_loop_success_thresholds.png")

# 2. Final Error (lower is better)
top_error = select_top_with_mpc(df, 'Final Error (m)', n=5, ascending=True)
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
fig.legend(handles=legend_elements, loc='upper right')
fig.savefig('results/closed_loop/plots/mean/closed_loop_final_error.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/closed_loop/plots/mean/closed_loop_final_error.png")

# 3. Solve Time (lower is better)
top_time = select_top_with_mpc(df, 'Solve Time (ms)', n=5, ascending=True)
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
fig.legend(handles=legend_elements, loc='upper right')
fig.savefig('results/closed_loop/plots/mean/closed_loop_solvetime.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/closed_loop/plots/mean/closed_loop_solvetime.png")

# 4. CPU Percent (lower is better)
top_cpu = select_top_with_mpc(df, 'CPU Percent', n=5, ascending=True)
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
fig.legend(handles=legend_elements, loc='upper right')
fig.savefig('results/closed_loop/plots/mean/closed_loop_cpu.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/closed_loop/plots/mean/closed_loop_cpu.png")

# 4b. Wall Time (lower is better)
top_walltime = select_top_with_mpc(df, 'Wall Time (s)', n=5, ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(range(len(top_walltime)), top_walltime['Wall Time (s)'], color=color_map(top_walltime['Type']))
ax.set_xticks(range(len(top_walltime)))
ax.set_xticklabels(top_walltime['Model'], rotation=45, ha='right')
ax.set_ylabel('Wall Time per Episode (s)')
ax.set_ylim([0, top_walltime['Wall Time (s)'].max() + 0.2])
ax.set_title('Top 5 - Episode Speed (Lower Better)', fontweight='bold')
for i, (idx, row) in enumerate(top_walltime.iterrows()):
    ax.text(i, row['Wall Time (s)'] + 0.02, f"{row['Wall Time (s)']:.3f}s", ha='center', va='bottom')
ax.grid(axis='y', alpha=0.3)
fig.legend(handles=legend_elements, loc='upper right')
fig.savefig('results/closed_loop/plots/mean/closed_loop_walltime.png', dpi=150, bbox_inches='tight')
print("Plot saved to results/closed_loop/plots/mean/closed_loop_walltime.png")

# 5. Steps to Success (all thresholds, lower is better)
top_steps_success = select_top_with_mpc(df.replace(0, np.nan).dropna(subset=['Steps to Success @0.02m']), 'Steps to Success @0.02m', n=5, ascending=True)
if not top_steps_success.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(top_steps_success))
    width = 0.25
    ax.bar(x_pos - width, top_steps_success['Steps to Success @0.02m'], width, label='@0.02m', color='#e74c3c')
    ax.bar(x_pos, top_steps_success['Steps to Success @0.03m'], width, label='@0.03m', color='#f39c12')
    ax.bar(x_pos + width, top_steps_success['Steps to Success @0.05m'], width, label='@0.05m', color='#27ae60')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_steps_success['Model'], rotation=45, ha='right')
    ax.set_ylabel('Mean Steps to Success')
    ax.set_title('Top 5 - Steps to Success (All Thresholds, Lower Better)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/closed_loop/plots/mean/closed_loop_steps_success_thresholds.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/closed_loop/plots/mean/closed_loop_steps_success_thresholds.png")

# 6. Steps (All Episodes) (lower is better)
top_steps_all = select_top_with_mpc(df.replace(0, np.nan).dropna(subset=['Steps (All Episodes)']), 'Steps (All Episodes)', n=5, ascending=True)
if not top_steps_all.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(top_steps_all)), top_steps_all['Steps (All Episodes)'], color=color_map(top_steps_all['Type']))
    ax.set_xticks(range(len(top_steps_all)))
    ax.set_xticklabels(top_steps_all['Model'], rotation=45, ha='right')
    ax.set_ylabel('Mean Steps (All Episodes)')
    ax.set_ylim([0, top_steps_all['Steps (All Episodes)'].max() + 5])
    ax.set_title('Top 5 - Steps (All Episodes) (Lower Better)', fontweight='bold')
    for i, (idx, row) in enumerate(top_steps_all.iterrows()):
        ax.text(i, row['Steps (All Episodes)'] + 0.5, f"{row['Steps (All Episodes)']:.1f}", ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.3)
    fig.legend(handles=legend_elements, loc='upper right')
    fig.savefig('results/closed_loop/plots/mean/closed_loop_steps_all.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/closed_loop/plots/mean/closed_loop_steps_all.png")

# 7. Tracking Error (lower is better)
top_track = select_top_with_mpc(df.replace(0, np.nan).dropna(subset=['Tracking Error (m)']), 'Tracking Error (m)', n=5, ascending=True)
if not top_track.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(top_track)), top_track['Tracking Error (m)'], color=color_map(top_track['Type']))
    ax.set_xticks(range(len(top_track)))
    ax.set_xticklabels(top_track['Model'], rotation=45, ha='right')
    ax.set_ylabel('Mean Tracking Error (m)')
    ax.set_ylim([0, top_track['Tracking Error (m)'].max() + 0.1])
    ax.set_title('Top 5 - Tracking Error (Lower Better)', fontweight='bold')
    for i, (idx, row) in enumerate(top_track.iterrows()):
        ax.text(i, row['Tracking Error (m)'] + 0.01, f"{row['Tracking Error (m)']:.3f}m", ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.3)
    fig.legend(handles=legend_elements, loc='upper right')
    fig.savefig('results/closed_loop/plots/mean/closed_loop_tracking_error.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/closed_loop/plots/mean/closed_loop_tracking_error.png")

# 8. Control Effort (lower is better)
top_effort = select_top_with_mpc(df.replace(0, np.nan).dropna(subset=['Control Effort']), 'Control Effort', n=5, ascending=True)
if not top_effort.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(top_effort)), top_effort['Control Effort'], color=color_map(top_effort['Type']))
    ax.set_xticks(range(len(top_effort)))
    ax.set_xticklabels(top_effort['Model'], rotation=45, ha='right')
    ax.set_ylabel('Mean Control Effort (||tau||)')
    ax.set_ylim([0, top_effort['Control Effort'].max() + 0.5])
    ax.set_title('Top 5 - Control Effort (Lower Better)', fontweight='bold')
    for i, (idx, row) in enumerate(top_effort.iterrows()):
        ax.text(i, row['Control Effort'] + 0.05, f"{row['Control Effort']:.2f}", ha='center', va='bottom')
    ax.grid(axis='y', alpha=0.3)
    fig.legend(handles=legend_elements, loc='upper right')
    fig.savefig('results/closed_loop/plots/mean/closed_loop_control_effort.png', dpi=150, bbox_inches='tight')
    print("Plot saved to results/closed_loop/plots/mean/closed_loop_control_effort.png")

# Print summary table
print("\n" + "="*100)
print("SUMMARY: Top 5 Models by Metric")
print("="*100)

print("\nSUCCESS RATE @ 0.02m (strict) - Higher is Better")
success_summary = top_success[['Model', 'Type', 'Success Rate @0.02m (%)']].copy()
print(success_summary.to_string(index=False))

print("\nSUCCESS RATE @ 0.03m (moderate) - Higher is Better")
success_mod = top_success[['Model', 'Type', 'Success Rate @0.03m']].copy()
success_mod['Success Rate @0.03m'] = success_mod['Success Rate @0.03m'] * 100
success_mod.columns = ['Model', 'Type', 'Success Rate @0.03m (%)']
print(success_mod.to_string(index=False))

print("\nSUCCESS RATE @ 0.05m (relaxed) - Higher is Better")
success_relaxed = top_success[['Model', 'Type', 'Success Rate @0.05m']].copy()
success_relaxed['Success Rate @0.05m'] = success_relaxed['Success Rate @0.05m'] * 100
success_relaxed.columns = ['Model', 'Type', 'Success Rate @0.05m (%)']
print(success_relaxed.to_string(index=False))

print("\nFINAL ERROR (Lower is Better)")
print(top_error[['Model', 'Type', 'Final Error (m)']].to_string(index=False))

print("\nSOLVE TIME (Lower is Better)")
print(top_time[['Model', 'Type', 'Solve Time (ms)']].to_string(index=False))

print("\nWALL TIME PER EPISODE (Lower is Better)")
print(top_walltime[['Model', 'Type', 'Wall Time (s)']].to_string(index=False))

print("\nCPU UTILIZATION (Lower is Better)")
print(top_cpu[['Model', 'Type', 'CPU Percent']].to_string(index=False))
