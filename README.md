# Hybrid Cloud Auto-Scaling with NF-DiagScale and Heterogeneous Containers

This repository implements NF-DiagScale (Neuro-Fuzzy Diagonal Scaler) and integrates a comprehensive heterogeneous container simulation environment based on the HGraphScale paper.

> **Naming note.** The Python package directory is still called `nfg_diagscale/` (and the controller class `NFGDiagScaleController`) — the legacy `nfg_` prefix is retained for import stability. The architecture's canonical name is **NF-DiagScale**; the original genetic (NSGA-II) magnitude planner has been replaced by a *deterministic exact* queue-model sizer (see Feature 3).

---

## Key Features

1. **Kalman + Holt Forecaster (+ GNN residual corrector)**: Per-microservice request-rate prediction using a Kalman filter (Kalman 1960) for RPS state estimation and Holt linear-trend exponential smoothing (Holt 1957) for the short-horizon ramp. A graph-aware residual corrector (a two-layer GCN, Kipf & Welling 2017) then adds a DAG-propagation correction learned from upstream load (`nfg_diagscale/hgraph_policy/gnn_forecast.py`); its labels are free (the realised next-interval load) and with no trained weights it is a no-op that falls back to the raw Kalman+Holt forecast. Retrain the shipped weights with `python train_forecast.py`.
2. **Adaptive ANFIS Controller**: A zero-order (singleton-consequent) Takagi-Sugeno neuro-fuzzy engine in the five-layer ANFIS arrangement (Jang 1993) that maps resource state and SLA risks into diagonal scaling instructions. Unlike a frozen inference engine, its rule consequents **self-tune online** from the realised SLO/cost outcome of the previous decision (direct adaptive fuzzy control; Wang 1993, MIT rule). The premise membership functions stay fixed so the fuzzy partition remains interpretable.
3. **Deterministic Exact Magnitude Sizer**: A reproducible feedforward sizer that exhaustively enumerates the bounded `(replicas, vCPU)` grid and returns the globally-optimal, cost-feasible knee `(h*, c*)` (STAR Eq. 8) — minimizing predicted response time subject to the budget, independent of any RNG. It still exposes the non-dominated (Pareto) latency/cost/rebalance trade-off front for reporting. *(This replaces the earlier NSGA-II genetic planner.)*
4. **M/D/1 Queue Latency Model**: Closed-form per-microservice latency (Kleinrock 1975) that replaces NF-DiagScale's original Themis look-up table.
5. **Heterogeneous Container Simulation**: The HGraphScale environment with Physical Machines, Virtual Machines, Best-Fit placement heuristics, and Capacity-Weighted load distribution.

---

## Control Pipeline

NF-DiagScale emits one scaling decision per 3-minute control interval through a six-stage closed loop:

1. **Forecast (F)** — a Kalman+Holt forecaster predicts each microservice's next-interval request count from `workload_his`, and a graph-aware GCN residual corrector adds a DAG-propagation correction from upstream load (no-op fallback to Kalman+Holt when no weights are present).
2. **Fuzzify (F)** — four grounded inputs per microservice: load pressure `psi` (CWRR-weighted batch-drain time / deadline), SLO headroom `omega`, cost headroom `phi` (remaining budget fraction), and a binary risk flag `rho`. The DAG upward rank (HEFT; Topcuoglu 2002) weights pressure toward critical-path microservices.
3. **Size** — the deterministic exact sizer returns the cost-feasible knee `(h*, c*)` that anchors the decision magnitude.
4. **Decide (N)** — the adaptive ANFIS blends the deterministic anchor with its fuzzy output into `(mode, delta_c, delta_n)`.
5. **Actuate (Diagonal)** — the target vCPU change is applied to the hottest replica; the simulator fills vertical headroom first and overflows into a new replica (vertical-first diagonal scaling). A budget guard blocks new-VM spawns that would breach the daily budget, keeping cost violation at zero.
6. **Learn (online)** — at the next interval the controller reads the realised response time and cumulative cost, forms the SLO and budget-pacing errors, and adapts the fired ANFIS rules' singleton consequents.

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
`run_star_comparison.py` drives the NF-DiagScale controller through the HGraphScale
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
  hgraph_policy/     # controller, Holt forecaster, deterministic exact sizer, M/D/1 queue model
  decision/          # adaptive ANFIS engine + fuzzy rule base (Jang 1993)
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
