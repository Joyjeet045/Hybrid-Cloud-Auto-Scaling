# Hybrid Cloud Auto-Scaling with NFG-DiagScale and Heterogeneous Containers

This repository implements NFG-DiagScale (Neuro-Fuzzy-Genetic Diagonal Auto-Scaler) and integrates a comprehensive heterogeneous container simulation environment based on the HGraphScale paper.

---

## Key Features

1. **Prophet-LSTM Hybrid Predictor**: Fast workload prediction with Prophet trend capture and LSTM residual refinement.
2. **ANFIS Controller**: A neuro-fuzzy engine mapping resource state and SLA risks directly into scaling instructions.
3. **NSGA-II Genetic Trajectory Planner**: Performs multi-objective optimization (minimizing SLA violations, instance cost, and rebalance actions) over a rolling forecast horizon.
4. **Heterogeneous Container Simulation**: Fully simulates Physical Machines, Virtual Machines, Best-Fit placement heuristics, and Capacity-Weighted load distribution.

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
Install all project dependencies inside a local virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Verify with Smoke Tests
Run the smoke test suite to check that the simulation, load balancer, cost models, and decision engines are fully operational:
```bash
python test_smoke.py
```

### 3. Run End-to-End Evaluation
Run the pipeline to train predictors, evaluate all baselines, and generate metrics and plots. Use the `--max-steps` parameter to restrict the test set length for rapid validation:
```bash
python main.py --max-steps 500
```
All comparative metric charts and JSON summaries will be output to the `results/` folder.
