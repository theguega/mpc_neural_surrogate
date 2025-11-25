"""
Compare closed-loop evaluation results from multiple controllers.

This script loads JSON results from closed-loop evaluations and creates
comparison tables and plots.
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(result_files):
    """Load results from JSON files."""
    results = []
    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                results.append({
                    'filepath': filepath,
                    'filename': os.path.basename(filepath),
                    'controller': data['controller_type'],
                    'model_path': data.get('model_path', 'N/A'),
                    'metrics': data['aggregate_metrics'],
                    'episodes': data['episode_metrics']
                })
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    return results


def print_comparison_table(results):
    """Print a comparison table of all results."""
    print("\n" + "=" * 140)
    print("Closed-Loop Evaluation Comparison")
    print("=" * 140)
    
    # Header
    header = (
        f"{'Controller':<20} | {'Success Rate':<15} | {'Mean Steps':<12} | "
        f"{'Final Error':<15} | {'Track Error':<15} | {'Control Effort':<15} | {'Solve Time (ms)':<15}"
    )
    print(header)
    print("-" * 140)
    
    for result in results:
        metrics = result['metrics']
        controller_name = result['controller']
        
        if result['model_path'] is not None and result['model_path'] != 'N/A':
            model_name = os.path.splitext(os.path.basename(result['model_path']))[0]
            controller_name = f"{controller_name} ({model_name[:15]})"
        
        success_rate = metrics['success_rate'] * 100
        mean_steps = metrics.get('mean_steps_to_target', None)
        final_error = metrics['mean_final_ee_error']
        final_error_std = metrics['std_final_ee_error']
        track_error = metrics['mean_tracking_error']
        track_error_std = metrics['std_tracking_error']
        control_effort = metrics['mean_control_effort']
        control_effort_std = metrics['std_control_effort']
        solve_time = metrics['mean_solve_time'] * 1000  # Convert to ms
        solve_time_std = metrics['std_solve_time'] * 1000
        
        # Format strings
        success_str = f"{success_rate:.1f}%"
        steps_str = f"{mean_steps:.1f}" if mean_steps is not None and mean_steps > 0 else "N/A"
        final_err_str = f"{final_error:.4f}±{final_error_std:.4f}"
        track_err_str = f"{track_error:.4f}±{track_error_std:.4f}"
        control_str = f"{control_effort:.2f}±{control_effort_std:.2f}"
        time_str = f"{solve_time:.2f}±{solve_time_std:.2f}"
        
        print(
            f"{controller_name:<20} | {success_str:<15} | {steps_str:<12} | "
            f"{final_err_str:<15} | {track_err_str:<15} | {control_str:<15} | {time_str:<15}"
        )
    
    print("=" * 140)


def plot_comparison(results, output_dir='results/closed_loop/plots'):
    """Create comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    names = []
    success_rates = []
    final_errors = []
    final_errors_std = []
    tracking_errors = []
    tracking_errors_std = []
    control_efforts = []
    control_efforts_std = []
    solve_times = []
    solve_times_std = []
    
    for result in results:
        metrics = result['metrics']
        controller_name = result['controller']
        
        if result['model_path'] is not None and result['model_path'] != 'N/A':
            model_name = os.path.splitext(os.path.basename(result['model_path']))[0]
            controller_name = f"{controller_name}\n({model_name[:15]})"
        
        names.append(controller_name)
        success_rates.append(metrics['success_rate'] * 100)
        final_errors.append(metrics['mean_final_ee_error'])
        final_errors_std.append(metrics['std_final_ee_error'])
        tracking_errors.append(metrics['mean_tracking_error'])
        tracking_errors_std.append(metrics['std_tracking_error'])
        control_efforts.append(metrics['mean_control_effort'])
        control_efforts_std.append(metrics['std_control_effort'])
        solve_times.append(metrics['mean_solve_time'] * 1000)
        solve_times_std.append(metrics['std_solve_time'] * 1000)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Closed-Loop Controller Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(names))
    width = 0.6
    
    # Success Rate
    ax = axes[0, 0]
    bars = ax.bar(x, success_rates, width, color='steelblue')
    ax.set_ylabel('Success Rate (%)', fontweight='bold')
    ax.set_title('Target Reaching Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Final Error
    ax = axes[0, 1]
    ax.bar(x, final_errors, width, yerr=final_errors_std, color='coral', capsize=5)
    ax.set_ylabel('Final EE Error (m)', fontweight='bold')
    ax.set_title('Final End-Effector Error')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Tracking Error
    ax = axes[0, 2]
    ax.bar(x, tracking_errors, width, yerr=tracking_errors_std, color='lightgreen', capsize=5)
    ax.set_ylabel('Mean Tracking Error (m)', fontweight='bold')
    ax.set_title('Average Tracking Error During Episode')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Control Effort
    ax = axes[1, 0]
    ax.bar(x, control_efforts, width, yerr=control_efforts_std, color='mediumpurple', capsize=5)
    ax.set_ylabel('Mean Control Effort', fontweight='bold')
    ax.set_title('Average Torque Magnitude')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Solve Time
    ax = axes[1, 1]
    ax.bar(x, solve_times, width, yerr=solve_times_std, color='gold', capsize=5)
    ax.set_ylabel('Solve Time (ms)', fontweight='bold')
    ax.set_title('Average Computation Time')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Speedup (relative to MPC)
    ax = axes[1, 2]
    if len(solve_times) > 0 and 'mpc' in names[0].lower():
        mpc_time = solve_times[0]
        speedups = [mpc_time / t if t > 0 else 0 for t in solve_times]
        bars = ax.bar(x, speedups, width, color='tomato')
        ax.set_ylabel('Speedup Factor', fontweight='bold')
        ax.set_title('Speedup Relative to MPC')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}x', ha='center', va='bottom', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'N/A\n(MPC baseline needed)', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'controller_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare closed-loop evaluation results"
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='results/closed_loop',
        help="Directory containing result JSON files (default: results/closed_loop)"
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help="Specific result files to compare (overrides --results-dir)"
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help="Generate comparison plots"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/closed_loop/plots',
        help="Directory to save plots (default: results/closed_loop/plots)"
    )
    
    args = parser.parse_args()
    
    # Load results
    if args.files:
        result_files = args.files
    else:
        pattern = os.path.join(args.results_dir, 'closed_loop_*.json')
        result_files = sorted(glob.glob(pattern))
    
    if not result_files:
        print(f"No result files found in {args.results_dir}")
        print("Run closed_loop_eval.py first to generate results.")
        return
    
    print(f"Found {len(result_files)} result file(s)")
    
    results = load_results(result_files)
    
    if not results:
        print("No valid results loaded")
        return
    
    # Print comparison table
    print_comparison_table(results)
    
    # Generate plots if requested
    if args.plot:
        try:
            plot_comparison(results, args.output_dir)
        except Exception as e:
            print(f"\nWarning: Could not generate plots: {e}")
            print("Make sure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    main()
