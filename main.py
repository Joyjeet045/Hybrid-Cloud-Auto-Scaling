"""
NFG-DiagScale: Neuro-Fuzzy-Genetic Diagonal Auto-Scaler
=======================================================

End-to-end evaluation pipeline.

This script:
  1. Loads real datasets (NASA HTTP / FIFA World Cup 1998)  [P5 sect 4.1]
  2. Trains Prophet-LSTM hybrid predictor                    [P5 sect 3.1]
  3. Runs NFG-DiagScale (MAPE-K + ANFIS + NSGA-II)         [P4, Jang93, P3]
  4. Runs baselines (HPA, VPA, DiagonalScale)               [P4 sect 5, P3]
  5. Computes KPIs and generates comparison plots            [P5 Eq. 10-13]

Usage:
  python main.py                       # NASA dataset (default)
  python main.py --dataset fifa_synthetic   # FIFA-like synthetic trace
  python main.py --dataset path/to/data.csv # Custom CSV with ds, y columns
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nfg_diagscale.config import load_config
from nfg_diagscale.data.loader import load_dataset
from nfg_diagscale.forecasting.hybrid_predictor import HybridPredictor
from nfg_diagscale.orchestrator import NFGDiagScaleOrchestrator, BaselineRunner
from nfg_diagscale.baselines.hpa import HPABaseline
from nfg_diagscale.baselines.vpa import VPABaseline
from nfg_diagscale.baselines.diagonal_scale import DiagonalScaleBaseline
from nfg_diagscale.baselines.themis import ThemisBaseline
from nfg_diagscale.evaluation.metrics import (
    compute_all_metrics, forecast_mape, forecast_rmse, forecast_mae, forecast_r2
)
from nfg_diagscale.evaluation.visualizer import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description="NFG-DiagScale Evaluation")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name: 'nasa', 'fifa_synthetic', or path to CSV")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for plots and metrics")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Limit test steps for quick runs")
    args = parser.parse_args()

    # ── Load configuration ──
    config = load_config(args.config)
    if args.dataset:
        config["dataset"]["name"] = args.dataset

    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("NFG-DiagScale: Neuro-Fuzzy-Genetic Diagonal Auto-Scaler")
    print("=" * 70)

    # ── Step 1: Load dataset [P5 sect 4.1] ──
    print("\n[Step 1] Loading dataset...")
    train_df, test_df = load_dataset(config)

    if args.max_steps and args.max_steps < len(test_df):
        test_df = test_df.iloc[:args.max_steps].copy()
        print(f"[Step 1] Limited test set to {args.max_steps} steps")

    # ── Step 2: Train Prophet-LSTM [P5 sect 3.1] ──
    print("\n[Step 2] Training Prophet-LSTM hybrid predictor...")
    t0 = time.time()
    predictor = HybridPredictor(config)
    predictor.train(train_df)
    print(f"[Step 2] Training complete in {time.time()-t0:.1f}s")

    # ── Step 2b: Evaluate forecast accuracy [P5 Eq. 10-13] ──
    print("\n[Step 2b] Evaluating forecast accuracy on test set...")
    predictions, prophet_pred, lstm_pred = predictor.predict_batch(test_df)
    actual = test_df["y"].values
    n = min(len(actual), len(predictions))

    print(f"  RMSE: {forecast_rmse(actual[:n], predictions[:n]):.2f}")
    print(f"  MAE:  {forecast_mae(actual[:n], predictions[:n]):.2f}")
    print(f"  MAPE: {forecast_mape(actual[:n], predictions[:n]):.2f}%")
    print(f"  R^2:  {forecast_r2(actual[:n], predictions[:n]):.4f}")

    # ── Step 3: Run NFG-DiagScale [P4 MAPE-K + Jang93 ANFIS + P3 NSGA-II] ──
    print("\n[Step 3] Running NFG-DiagScale...")
    t0 = time.time()
    nfg = NFGDiagScaleOrchestrator(config, predictor)
    nfg_history, nfg_actions = nfg.run_evaluation(test_df)
    print(f"[Step 3] NFG-DiagScale complete in {time.time()-t0:.1f}s")

    # ── Step 4: Run baselines ──
    print("\n[Step 4] Running baselines...")
    all_results = {"NFG-DiagScale": {"history": nfg_history, "actions": nfg_actions}}
    baselines = [
        HPABaseline(config),
        VPABaseline(config),
        ThemisBaseline(config),
        DiagonalScaleBaseline(config),
    ]

    for baseline in baselines:
        print(f"  Running {baseline.name}...")
        t0 = time.time()
        runner = BaselineRunner(config, baseline)
        history, actions = runner.run_evaluation(test_df)
        all_results[baseline.name] = {"history": history, "actions": actions}
        print(f"  {baseline.name} complete in {time.time()-t0:.1f}s")

    # ── Step 5: Compute KPIs ──
    print("\n[Step 5] Computing KPIs...")
    slo = config["themis"]["slo_ms"]
    all_metrics = []

    for name, result in all_results.items():
        m = compute_all_metrics(result["history"], result["actions"], slo, name)
        all_metrics.append(m)

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Autoscaler':<20} {'SVR%':>8} {'Cost($)':>10} {'AvgLat':>10} "
          f"{'P99Lat':>10} {'Actions':>8} {'Rebal':>8}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['name']:<20} {m['svr_pct']:>8.2f} {m['total_cost']:>10.2f} "
              f"{m['avg_latency_ms']:>10.2f} {m['p99_latency_ms']:>10.2f} "
              f"{m['scaling_actions']:>8d} {m['rebalance_overhead']:>8d}")
    print("=" * 80)

    # Cost efficiency ratios
    nfg_cost = all_metrics[0]["total_cost"]
    print("\nCost Efficiency Ratios (vs NFG-DiagScale):")
    for m in all_metrics[1:]:
        cer = m["total_cost"] / max(nfg_cost, 0.01)
        print(f"  {m['name']}: {cer:.2f}x")

    # ── Step 6: Generate plots ──
    print(f"\n[Step 6] Generating plots to {output_dir}...")
    generate_all_plots(
        all_results, all_metrics, slo,
        test_df, predictions, output_dir
    )

    import json
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n[DONE] All results saved.")
    return all_metrics


if __name__ == "__main__":
    main()
