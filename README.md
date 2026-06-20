# Hybrid Cloud Auto-Scaling with NFG-DiagScale and Heterogeneous Containers

This repository implements NFG-DiagScale (Neuro-Fuzzy-Genetic Diagonal Auto-Scaler) and integrates a comprehensive heterogeneous container simulation environment based on the HGraphScale paper.

---

## Key Features

1. **Kalman + Holt Forecaster**: Per-container-type request-rate prediction using a Kalman filter (Kalman 1960) for RPS state estimation and Holt linear-trend exponential smoothing (Holt 1957) for the short-horizon ramp.
2. **ANFIS Controller**: An inference-mode Takagi-Sugeno neuro-fuzzy engine (Jang 1993) mapping resource state and SLA risks directly into diagonal scaling instructions.
3. **NSGA-II Genetic Trajectory Planner**: Multi-objective optimization (Deb et al. 2002) over the (replicas, cores) plane, minimizing SLA violations, instance cost, and rebalance actions across a rolling forecast horizon.
4. **M/D/1 Queue Latency Model**: Closed-form per-microservice latency (Kleinrock 1975) that replaces NFG-DiagScale's original Themis look-up table.
5. **Heterogeneous Container Simulation**: The HGraphScale environment with Physical Machines, Virtual Machines, Best-Fit placement heuristics, and Capacity-Weighted load distribution.

---

## Heterogeneous Simulation Environment (HGraphScale Model)

The simulator implements the exact heterogeneous deployment specs from Table III of the HGraphScale paper:

### VM Types & Hourly Pricing
* `m5.xlarge`: 4 vCPUs, 16 GiB RAM ($0.192/hour)
* `m5.2xlarge`: 8 vCPUs, 32 GiB RAM ($0.384/hour)
* `m5.4xlarge`: 16 vCPUs, 64 GiB RAM ($0.768/hour)
* `m5.8xlarge`: 32 vCPUs, 128 GiB RAM ($1.536/hour)
* `m5.12xlarge`: 48 vCPUs, 192 GiB RAM ($2.304/hour)

### Core Integration Details
- **PM Constraints**: Each Physical Machine (PM) has a fixed capacity limit of **64 vCPUs**.
- **Best-Fit Placement**: When new containers or VMs are requested, the environment maps them to the existing nodes with the smallest sufficient leftover capacity (Best-Fit Decreasing heuristic) to minimize resource fragmentation.
- **Hierarchical Scale-Up/Down Executor**: Scale-up actions perform vertical scaling first. If host VM bounds are hit, it scales up to the instance boundary and spawns new containers (horizontal scale-out) with the remainder. Scale-down actions consolidate and shutdown empty VMs/PMs.
- **Capacity-Weighted Round-Robin (CWRR)**: Incoming traffic is distributed among container replicas based on their allocated cores. The overall service response latency is calculated as a capacity-weighted average.
- **Rental Cost KPI**: Computes actual infrastructure expenses based on leased instance hourly prices rather than static per-core assumptions.

---

## Usage Instructions

### 1. Setup Virtual Environment
Install dependencies inside a local virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Reproduce the STAR Comparison
`run_star_comparison.py` drives the NFG-DiagScale controller through the HGraphScale
environment on the nine STAR benchmark scenarios (NASA / Wikipedia / Alibaba x A11 / A12 / A13)
and tabulates mean response time (MRT) and SLA-violation cost against STAR's reported Table 3:
```bash
# All nine scenarios:
python run_star_comparison.py --seeds 0

# A single scenario written to its own JSON:
python run_star_comparison.py --scenario A-12 --seeds 0 --out results_a12.json
```
Each scenario reports measured `MRT` (ms) and `Vio = max(0, cost - $200)` so results can be
compared directly with the published STAR baselines (AWS-Scale, ProScale, DeepScale, DRPC, STAR).

### 3. App-14 (beyond STAR Table 3)
STAR Table 3 stops at the 13-microservice app. The 14-microservice scenarios are
benchmarked separately against HGraphScale (IEEE TSC) Table IV:
```bash
python run_star_comparison.py --scenario N-14 W-14 A-14 --seeds 0 --out star_comparison_results.json
```

### 4. Generate the report figures and tables
`reporting/make_report.py` renders every figure and table from the measured
`star_comparison_results.json` plus one instrumented episode (requires `matplotlib`):
```bash
python reporting/make_report.py --rep A-12
```
Outputs are written to `reporting/figures/` (fig1-fig10) and `reporting/tables/`
(table1-table5, each as `.md` and `.csv`).

---

## Repository Layout
```
nfg_diagscale/
  config/            # default.yaml + loader (all hyperparameters)
  forecasting/       # Kalman filter (Kalman 1960)
  hgraph_policy/     # controller, Holt forecaster, NSGA-II optimizer, M/D/1 queue model
  decision/          # ANFIS engine + fuzzy rule base (Jang 1993)
  hgraph_env/        # state adapter + vendored HGraphScale simulator under env/
    env/autoscaling_v1/
      dax/           # application DAGs (App_11..App_14.xml)
      lib/           # simulator core + alibaba_workload.json (real v2022 trace)
run_star_comparison.py        # 9 STAR scenarios + 3 App-14 scenarios
reporting/make_report.py      # figures + tables
star_comparison_results.json  # measured results (12 scenarios)
requirements.txt
```

> **Note on workloads.** NASA and Wikipedia request traces are embedded directly
> in `lib/simsetting.py`; the Alibaba trace is derived from the real
> cluster-trace-microservices-v2022 (Luo et al., 2022) and stored as
> `lib/alibaba_workload.json`. No additional data download is required to
> reproduce the 12 reported scenarios.
