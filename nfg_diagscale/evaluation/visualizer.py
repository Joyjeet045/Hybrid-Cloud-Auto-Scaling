"""
Visualization module for autoscaler evaluation results.

Aesthetics: Premium dark theme with research-grade styling.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set premium aesthetics
plt.style.use('dark_background')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Arial']
matplotlib.rcParams['axes.facecolor'] = '#121212'
matplotlib.rcParams['figure.facecolor'] = '#0a0a0a'
matplotlib.rcParams['grid.color'] = '#333333'
matplotlib.rcParams['axes.edgecolor'] = '#444444'
matplotlib.use("Agg")

COLORS = [
    '#00d4ff', # Electric Blue (NFG-DiagScale)
    '#ff4d4d', # Vibrant Red
    '#ffaa00', # Amber
    '#00ff88', # Spring Green
    '#bb86fc', # Soft Purple
    '#cfd8dc'  # Blue Grey
]


def plot_workload_and_predictions(test_df, predictions, save_dir):
    """Plot actual workload vs Prophet-LSTM predictions."""
    fig, ax = plt.subplots(figsize=(15, 6))
    actual = test_df["y"].values
    n = min(len(actual), len(predictions))
    t = np.arange(n)

    ax.plot(t, actual[:n], label="Actual RPS", color=COLORS[5], alpha=0.4, linewidth=1.0)
    ax.plot(t, predictions[:n], label="Prophet-LSTM Hybrid Predicted", color=COLORS[0], alpha=0.9, linewidth=1.2)
    
    ax.set_xlabel("Time Step (minutes)", color='#888888', fontsize=12)
    ax.set_ylabel("Requests Per Minute", color='#888888', fontsize=12)
    ax.set_title("Workload Forecasting Fidelity", fontsize=16, fontweight='bold', pad=20)
    ax.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#444444')
    ax.grid(True, linestyle='--', alpha=0.2)
    
    # Remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "prediction_accuracy.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_latency_comparison(all_results, slo, save_dir):
    """Plot latency time series for all autoscalers."""
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (name, result) in enumerate(all_results.items()):
        history = result["history"]
        lats = [s["app_latency"] for s in history]
        t = np.arange(len(lats))

        # Downsample for readability by taking the window max to preserve spikes
        step_sz = max(1, len(lats) // 1500)
        if step_sz > 1:
            lats_ds = [np.max(lats[j:j+step_sz]) for j in range(0, len(lats), step_sz)]
            t_ds = t[::step_sz]
            ax.plot(t_ds, lats_ds, label=name, color=COLORS[i % len(COLORS)], alpha=0.8, linewidth=1.2 if name == "NFG-DiagScale" else 0.8)
        else:
            ax.plot(t, lats, label=name, color=COLORS[i % len(COLORS)], alpha=0.8, linewidth=1.2 if name == "NFG-DiagScale" else 0.8)

    ax.axhline(y=slo, color="#ff4d4d", linestyle="--", label=f"SLO Target ({slo}ms)", alpha=0.6, linewidth=1.5)
    ax.set_xlabel("Evaluation Time Steps", color='#888888', fontsize=12)
    ax.set_ylabel("Response Latency (ms)", color='#888888', fontsize=12)
    ax.set_title("Resiliency and SLO Compliance", fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', frameon=True, facecolor='#1e1e1e', edgecolor='#444444')
    ax.grid(True, linestyle='--', alpha=0.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "latency_comparison.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_cost_comparison(all_results, save_dir):
    """Plot cumulative cost over time for all autoscalers."""
    fig, ax = plt.subplots(figsize=(15, 6))

    for i, (name, result) in enumerate(all_results.items()):
        history = result["history"]
        costs = np.cumsum([s["step_cost"] for s in history])
        t = np.arange(len(costs))
        step = max(1, len(costs) // 1000)
        ax.plot(t[::step], costs[::step], label=name, color=COLORS[i % len(COLORS)], alpha=0.9, linewidth=2.0 if name == "NFG-DiagScale" else 1.2)

    ax.set_xlabel("Evaluation Time Steps", color='#888888', fontsize=12)
    ax.set_ylabel("Cumulative Infrastructure Cost ($)", color='#888888', fontsize=12)
    ax.set_title("Economic Efficiency Benchmarking", fontsize=16, fontweight='bold', pad=20)
    ax.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#444444')
    ax.grid(True, linestyle='--', alpha=0.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cost_comparison.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_scaling_trajectories(all_results, save_dir):
    """Plot both H and V trajectories in a multi-panel plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for i, (name, result) in enumerate(all_results.items()):
        history = result["history"]
        reps = [s["replicas"] for s in history]
        cores = [s["cores"] for s in history]
        t = np.arange(len(reps))
        step = max(1, len(reps) // 1500)
        
        ax1.step(t[::step], reps[::step], label=name, color=COLORS[i % len(COLORS)], alpha=0.8, linewidth=1.5, where="post")
        ax2.step(t[::step], cores[::step], label=name, color=COLORS[i % len(COLORS)], alpha=0.8, linewidth=1.5, where="post")

    ax1.set_ylabel("Replica Count (Horizontal)", color='#888888', fontsize=12)
    ax1.set_title("Horizontal Scaling Trajectories", fontsize=14, fontweight='bold')
    ax1.legend(frameon=True, facecolor='#1e1e1e', edgecolor='#444444')
    ax1.grid(True, linestyle='--', alpha=0.1)
    
    ax2.set_xlabel("Evaluation Time Steps", color='#888888', fontsize=12)
    ax2.set_ylabel("CPU Cores (Vertical)", color='#888888', fontsize=12)
    ax2.set_title("Vertical Resource Trajectories", fontsize=14, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.1)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "scaling_trajectories.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_kpi_summary(all_metrics, save_dir):
    """Premium KPI radar/bar summary."""
    names = [m["name"] for m in all_metrics]
    svr = [m["svr_pct"] for m in all_metrics]
    costs = [m["total_cost"] for m in all_metrics]
    rebal = [m["rebalance_overhead"] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. SLO Violations
    bars0 = axes[0].bar(names, svr, color=[COLORS[i % len(COLORS)] for i in range(len(names))], alpha=0.8)
    axes[0].set_ylabel("SLO Violation Rate (%)", color='#888888')
    axes[0].set_title("Service Reliability", fontsize=14, fontweight='bold')
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylim(0, max(5, max(svr) * 1.2 if svr else 5)) # Min height for scale
    
    # 2. Total Cost
    bars1 = axes[1].bar(names, costs, color=[COLORS[i % len(COLORS)] for i in range(len(names))], alpha=0.8)
    axes[1].set_ylabel("Total TCO ($)", color='#888888')
    axes[1].set_title("Cost Optimization", fontsize=14, fontweight='bold')
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylim(0, max(10, max(costs) * 1.2 if costs else 10))
    
    # 3. Stability (Rebalance Overhead)
    bars2 = axes[2].bar(names, rebal, color=[COLORS[i % len(COLORS)] for i in range(len(names))], alpha=0.8)
    axes[2].set_ylabel("Rebalance Overhead Score", color='#888888')
    axes[2].set_title("Operational Stability", fontsize=14, fontweight='bold')
    axes[2].tick_params(axis="x", rotation=35)
    axes[2].set_ylim(0, max(5, max(rebal) * 1.2 if rebal else 5))

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.2)

    fig.suptitle("NFG-DiagScale Multi-Metric Performance Benchmark", fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "kpi_benchmark.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_latency_distribution(all_results, save_dir):
    """Premium boxplot to show latency distribution and stability."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    data = []
    labels = []
    colors = []
    
    for i, (name, result) in enumerate(all_results.items()):
        lats = [s["app_latency"] for s in result["history"]]
        data.append(lats)
        labels.append(name)
        colors.append(COLORS[i % len(COLORS)])

    # Premium styling for boxes
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=True, 
                    showfliers=False, medianprops={'color': 'white', 'linewidth': 2})
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)

    ax.set_ylabel("Response Latency (ms)", color='#888888', fontsize=12)
    ax.set_title("Statistical Latency Stability", fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, axis='y', linestyle='--', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Custom x-axis rotation
    plt.xticks(rotation=25)
    
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "latency_distribution.png"), dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def generate_all_plots(all_results, all_metrics, slo, test_df, predictions, save_dir):
    """Generate all evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    if predictions is not None and test_df is not None:
        plot_workload_and_predictions(test_df, predictions, save_dir)

    plot_latency_comparison(all_results, slo, save_dir)
    plot_latency_distribution(all_results, save_dir)
    plot_cost_comparison(all_results, save_dir)
    plot_scaling_trajectories(all_results, save_dir)
    plot_kpi_summary(all_metrics, save_dir)

    print(f"[Visualizer] Research-grade plots generated at {save_dir}")
