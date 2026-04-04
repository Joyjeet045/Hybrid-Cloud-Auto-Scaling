"""
Visualization module for autoscaler evaluation results.

Generates comparison plots across all baselines and NFG-DiagScale.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def plot_workload_and_predictions(test_df, predictions, save_dir):
    """Plot actual workload vs Prophet-LSTM predictions."""
    fig, ax = plt.subplots(figsize=(14, 5))
    actual = test_df["y"].values
    n = min(len(actual), len(predictions))
    t = np.arange(n)

    ax.plot(t, actual[:n], label="Actual RPS", alpha=0.7, linewidth=0.8)
    ax.plot(t, predictions[:n], label="Prophet-LSTM Predicted", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Time Step (minutes)")
    ax.set_ylabel("Requests Per Minute")
    ax.set_title("Workload Prediction: Prophet-LSTM Hybrid Model")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "prediction_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_latency_comparison(all_results, slo, save_dir):
    """Plot latency time series for all autoscalers."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, result in all_results.items():
        history = result["history"]
        lats = [s["app_latency"] for s in history]
        t = np.arange(len(lats))

        # Downsample for readability
        step = max(1, len(lats) // 2000)
        ax.plot(t[::step], lats[::step], label=name, alpha=0.7, linewidth=0.8)

    ax.axhline(y=slo, color="red", linestyle="--", label=f"SLO = {slo}ms", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Comparison Across Autoscalers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "latency_comparison.png"), dpi=150)
    plt.close(fig)


def plot_cost_comparison(all_results, save_dir):
    """Plot cumulative cost over time for all autoscalers."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, result in all_results.items():
        history = result["history"]
        costs = np.cumsum([s["step_cost"] for s in history])
        t = np.arange(len(costs))
        step = max(1, len(costs) // 2000)
        ax.plot(t[::step], costs[::step], label=name, alpha=0.7, linewidth=1.0)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cumulative Cost ($)")
    ax.set_title("Infrastructure Cost Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cost_comparison.png"), dpi=150)
    plt.close(fig)


def plot_replica_trace(all_results, save_dir):
    """Plot replica count over time for all autoscalers."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, result in all_results.items():
        history = result["history"]
        reps = [s["replicas"] for s in history]
        t = np.arange(len(reps))
        step = max(1, len(reps) // 2000)
        ax.plot(t[::step], reps[::step], label=name, alpha=0.7, linewidth=1.0)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Replica Count")
    ax.set_title("Scaling Trajectory: Replica Count Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "replica_trace.png"), dpi=150)
    plt.close(fig)


def plot_cores_trace(all_results, save_dir):
    """Plot CPU cores allocation over time."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for name, result in all_results.items():
        history = result["history"]
        cores = [s["cores"] for s in history]
        t = np.arange(len(cores))
        step = max(1, len(cores) // 2000)
        ax.plot(t[::step], cores[::step], label=name, alpha=0.7, linewidth=1.0)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("CPU Cores Per Pod")
    ax.set_title("Scaling Trajectory: CPU Cores Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "cores_trace.png"), dpi=150)
    plt.close(fig)


def plot_kpi_bar_chart(all_metrics, save_dir):
    """Bar chart comparing KPIs across all autoscalers."""
    names = [m["name"] for m in all_metrics]
    svr = [m["svr_pct"] for m in all_metrics]
    costs = [m["total_cost"] for m in all_metrics]
    actions = [m["scaling_actions"] for m in all_metrics]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(names, svr, color=["#e74c3c" if v > 1 else "#2ecc71" for v in svr])
    axes[0].set_ylabel("SLO Violation Rate (%)")
    axes[0].set_title("SLO Violations")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, costs, color="#3498db")
    axes[1].set_ylabel("Total Cost ($)")
    axes[1].set_title("Infrastructure Cost")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(names, actions, color="#9b59b6")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Scaling Actions")
    axes[2].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "kpi_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(all_results, all_metrics, slo, test_df, predictions, save_dir):
    """Generate all evaluation plots."""
    os.makedirs(save_dir, exist_ok=True)

    if predictions is not None and test_df is not None:
        plot_workload_and_predictions(test_df, predictions, save_dir)

    plot_latency_comparison(all_results, slo, save_dir)
    plot_cost_comparison(all_results, save_dir)
    plot_replica_trace(all_results, save_dir)
    plot_cores_trace(all_results, save_dir)
    plot_kpi_bar_chart(all_metrics, save_dir)

    print(f"[Visualizer] All plots saved to {save_dir}")
