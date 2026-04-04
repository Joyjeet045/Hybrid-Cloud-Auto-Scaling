# NFG-DiagScale: A Neuro-Fuzzy-Genetic Diagonal Auto-Scaler for Cloud Systems

> **Research Proposal — Final Version (Post-Audit), April 2026**
> *Two-pass audit against actual published papers completed. All removed/corrected claims documented below.*

---

## PART I — AUDIT REPORT (What Was Wrong in v1)

This section documents every claim from the prior proposal that was hallucinated, overclaimed, or forced for integration optics, with the correction or removal decision for each.

---

### A-1 · Hallucinated Latency Equation (P1 — Themis)

**The prior formula in v1:**

$$q_s(b, c, n) = \max\!\left(\frac{b-1}{\hat{\lambda}_{t+k}},\ l_s(b, c) - \frac{nb+1}{\hat{\lambda}_{t+k}}\right)$$

**Verdict: ❌ REMOVED — Not verifiable from arXiv:2407.14843.**

Themis does **not** publish a closed-form queuing equation of this form. It uses **offline profiling** to build latency tables indexed by `(batch_size, CPU_cores)` and combines them with a DP/IP optimizer. The `max(...)` formula above is a hallucinated M/D/1 approximation that does not appear in the paper. No closed-form $L_{queue}$ is published.

**Correct use of Themis:** The principle that $L_{total}(b, c, n) \triangleq L_{profile}(b, c) + L_{queue}(\lambda, b, n)$, where $L_{profile}$ is extracted from an offline profiling table and $L_{queue}$ is estimated numerically by the DP/IP solver. No analytical queuing formula is attributed to this paper.

---

### A-2 · Overclaimed Detection Time Numbers (P4 — MAPE-K / AspectJ)

**The prior claim:** *"AspectJ reduces detection latency from ~120s (infra-only) to ~30s (app-layer) as demonstrated in [P4]."*

**Verdict: ❌ REMOVED — These specific numbers are fabricated.**

The ACM UCC'25 paper (DOI: 10.1145/3773274.3774256) makes the architectural point that application-layer probes surface signals before infrastructure saturation, but publishes no "120s vs 30s" experimental figures. The architectural benefit is retained qualitatively. All specific timing numbers are removed.

---

### A-3 · GNN-RaPP Integration (P2 — HAS-GPU) — Forced and Context-Mismatched

**The prior use of P2:** Cited HAS-GPU's GNN-based RaPP predictor as a core component.

**Verdict: ⚠️ DROPPED — GPU-specific, not transferable to general CPU-bound Kubernetes services.**

RaPP is a GNN trained specifically over GPU SM-partition + time-quota configurations. It cannot be transplanted into a CPU-centric general-purpose auto-scaler without abandoning its identity as RaPP from P2.

**What survives from P2:** The **Kalman Filter** for short-term RPS estimation — explicitly described in arXiv:2505.01968 §IV-A, general-purpose, and directly applicable. GNN-RaPP is dropped entirely.

---

### A-4 · Kalman Filter Misclassified as "Neural Network"

**Verdict: ❌ CORRECTED — A Kalman filter is a Bayesian state estimator, not a neural network.**

This misclassification inflated AI-technique diversity but would not survive peer review. The Kalman filter is re-classified as a signal-processing/Bayesian estimation module and does not count toward the N/F/G acronym.

---

### A-5 · "Vertical-First Heuristic" Misattribution to Themis (P1)

**Verdict: ⚠️ CORRECTED — Themis is a DP/IP optimizer, not a named heuristic.**

Themis's strategy is: react with immediate in-place vertical scaling (zero cold-start penalty), then transition to horizontal at steady state — optimized by DP/IP, not a standing priority rule. The ANFIS rules now express the *economic insight* from Themis without misattributing it as a named heuristic.

---

### A-6 · NSGA-II Trajectory Encoding Not Disclosed as Novel

**Verdict: ⚠️ CORRECTED — Must be explicitly stated as novel synthesis.**

No reference paper encodes a T-step diagonal scaling trajectory as a chromosome for NSGA-II. This is now explicitly flagged as the **primary novel contribution of Layer 3**.

---

### A-7 · Name Integrity After Kalman Re-classification

The NFG acronym is justified cleanly by:
- **N** = Prophet-LSTM Neural Forecaster (P5) ✅
- **F** = ANFIS Neuro-Fuzzy Decision Engine ✅
- **G** = NSGA-II Genetic Global Optimizer ✅

---

## PART II — FINAL PROPOSAL (All Claims Verified)

---

# NFG-DiagScale: A Neuro-Fuzzy-Genetic Diagonal Auto-Scaler for Cloud Systems

> **Short name:** NFG-DiagScale
> **N = Neural (Prophet-LSTM) | F = Fuzzy (ANFIS) | G = Genetic (NSGA-II)**

---

## 1. Problem Statement

Modern cloud autoscalers cannot simultaneously answer three questions with precision:

| Question | What the Right Answer Requires |
|---|---|
| *When* to scale? | Proactive, multi-timescale workload forecasting |
| *Which dimension* — horizontal (replicas) or vertical (per-replica resources)? | A decision engine reasoning over the 2D (H, V) Scaling Plane |
| *How much* to scale, globally over time? | A multi-objective optimizer considering cost, SLO risk, and transition overhead jointly |

Existing systems [P1–P5] solve at most two of these in isolation. **NFG-DiagScale** is the first framework to address all three using a layered, coherent integration of neural forecasting, neuro-fuzzy decision-making, and genetic trajectory optimization on the Diagonal Scaling Plane.

---

## 2. Reference Papers (Verified)

| # | Paper | Source | Year | Verified Use |
|---|---|---|---|---|
| **P1** | *A Tale of Two Scales: Reconciling Horizontal and Vertical Scaling for Inference Serving Systems* (Themis) | arXiv:2407.14843 | 2024 | Profiling-based latency surface $L_{profile}(b,c)$; in-place vertical → horizontal transition principle; DP/IP optimization |
| **P2** | *HAS-GPU: Efficient Hybrid Auto-scaling with Fine-grained GPU Allocation for SLO-aware Serverless Inferences* | arXiv:2505.01968 | 2025 | **Kalman Filter for RPS estimation only.** GNN-RaPP dropped (GPU-specific) |
| **P3** | *Diagonal Scaling: A Multi-Dimensional Resource Model and Optimization Framework for Distributed Databases* | arXiv:2511.21612 | 2025 | 2D Scaling Plane $(H,V)$; DIAGONALSCALE local-search; verified Rebalance Penalty formula |
| **P4** | *An Autonomic Computing Approach for Scaling Cloud-Based Smart City Platforms* | ACM UCC'25, DOI: 10.1145/3773274.3774256 | 2025 | MAPE-K control loop architecture; AspectJ multi-level monitoring; composite stress metric concept |
| **P5** | *Time Series Forecasting-based Kubernetes Autoscaling using Facebook Prophet and LSTM* | Frontiers in Computer Science, doi:10.3389/fcomp.2025.1509165 | 2025 | Hybrid Prophet-LSTM forecaster equations; proactive pod-count formula |

> **Classical foundations:** NSGA-II (Deb et al., 2002); ANFIS (Jang, 1993)

---

## 3. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    MAPE-K Autonomic Loop  [P4]                        │
│                                                                        │
│  MONITOR                   ANALYZE                                     │
│  ┌───────────────┐         ┌──────────────────────────────────────┐   │
│  │  AspectJ       │────────▶│  Layer 1: Prophet-LSTM Forecaster    │   │
│  │  (infra +      │         │  [P5]  +  Kalman RPS Smoother  [P2] │   │
│  │   app-layer)   │         └────────────────────┬─────────────────┘   │
│  │  [P4]          │                              │                      │
│  └───────────────┘                              ▼                      │
│                              ┌──────────────────────────────────────┐  │
│                              │  PLAN — Layer 2: ANFIS Engine         │  │
│                              │  Inputs: Ψ, Ω, Φ  [P5, P1, P3, P4]  │  │
│                              │  Output: (Mode, Δc, Δn)              │  │
│                              └──────────────────┬───────────────────┘  │
│                                                  │                      │
│  EXECUTE                                         ▼                      │
│  ┌───────────────┐         ┌──────────────────────────────────────┐   │
│  │  Vertical /    │◀────────│  Layer 3: NSGA-II Trajectory         │   │
│  │  Diagonal /    │         │  Optimizer  [P3 + NSGA-II]           │   │
│  │  Horizontal    │         │  Every 15 min → (H*,V*)_{τ=1..T}    │   │
│  └───────────────┘         └──────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1 — Neural Workload Forecaster [P5 + P2]

### 4.1 Hybrid Prophet-LSTM Forecaster (from [P5])

Guruge & Priyadarshana (2025) propose and validate this on the NASA HTTP and FIFA '98 datasets:

**Step A — Prophet trend and seasonality:**

$$\hat{\lambda}^{(P)}_t = g(t) + s(t) + h(t) + \epsilon_t$$

- $g(t)$: piecewise-linear growth trend fitted on historical RPS
- $s(t)$: Fourier-series seasonality (weekly + daily harmonics)
- $h(t)$: holiday/event effect terms
- $\epsilon_t \sim \mathcal{N}(0, \sigma^2)$: Gaussian noise

**Step B — LSTM residual correction:**

$$\hat{r}_{t+k} = \text{LSTM}\!\bigl([r_{t-\tau}, \ldots, r_t;\ \Theta_{LSTM}]\bigr)$$

where $r_t = \lambda_t - \hat{\lambda}^{(P)}_t$ captures non-linear anomalies and burst spikes.

**Step C — Fused prediction:**

$$\hat{\lambda}_{t+k} = \hat{\lambda}^{(P)}_{t+k} + \hat{r}_{t+k}$$

Prediction horizons $k \in \{1, 5, 15\}$ minutes.

**Proactive baseline pod count (from [P5]):**

$$n^{H}_{t+k} = \left\lceil \frac{\hat{\lambda}_{t+k}}{\lambda_{pod}^{max}} \right\rceil$$

This is Layer 1's horizontal suggestion that seeds the ANFIS decision engine.

### 4.2 Kalman Filter RPS Smoother (from [P2])

HAS-GPU (arXiv:2505.01968, §IV-A) uses a Kalman filter to produce a low-noise intra-window RPS estimate. Adopted directly as a pre-processing step:

$$\hat{\lambda}^{KF}_{t|t} = \hat{\lambda}^{KF}_{t|t-1} + K_t\!\left(\lambda^{obs}_t - \hat{\lambda}^{KF}_{t|t-1}\right)$$

$$K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$$

where $K_t$ is the Kalman gain, $P_{t|t-1}$ is the prior error covariance, and $R$ is the measurement noise variance.

> **Classification:** Kalman filter = Bayesian state estimator. It is a signal-processing module, not a neural network. It does not count toward the N/F/G acronym.

> **[Novel claim N1]:** P5 provides long-horizon planning with no noise suppression. P2's Kalman provides fast intra-window smoothing with no planning horizon. Neither combines them. This multi-timescale fusion is the first novelty contribution.

---

## 5. Layer 2 — ANFIS Diagonal Scaling Decision Engine [P1 + P3 + P4 + ANFIS]

### 5.1 Motivation for ANFIS

The 2D Scaling Plane $(H, V)$ from P3 has a discrete-continuous hybrid structure ($H$ is integer; $V$ is real-valued). The mapping from workload state to the correct scaling mode requires handling:

- Uncertainty in multi-step forecasts
- Competing soft constraints (SLO slack vs. cost vs. rebalance disruption)
- Context-dependence of which dimension is more efficient to scale

ANFIS (Jang, 1993) is the natural architecture: its **Mamdani fuzzy rule base** encodes interpretable expert knowledge; its **neural backprop/least-squares layer** tunes membership function parameters from historical data without manual calibration.

> **[Novel claim N2]:** No prior work uses ANFIS as the decision engine on P3's Diagonal Scaling Plane. P3 uses a greedy single-step local search; P4 uses fixed threshold rules.

### 5.2 Input Linguistic Variables

| Variable | Symbol | Physical Meaning | Range | Fuzzy Sets |
|---|---|---|---|---|
| Workload Surge Ratio | $\Psi = \hat{\lambda}_{t+k} / \lambda^{KF}_t$ | Forecasted excess over current smoothed load | [0, 5] | {Low, Moderate, High, Critical} |
| Latency Headroom | $\Omega = (SLO - L_{curr}) / SLO$ | Fraction of SLO budget currently unused | [0, 1] | {Tight, Comfortable, Ample} |
| Vertical Resource Headroom | $\Phi = 1 - (c_{curr} / c_{max})$ | Fraction of per-replica resource ceiling that is free | [0, 1] | {Exhausted, Available, Abundant} |

**Source tracing:**
- $\Psi$: from Layer 1 forecast [P5] and Kalman-smoothed current load [P2]
- $\Omega$: from AspectJ app-layer latency probes [P4] against SLO target [P1]
- $\Phi$: from current position on the vertical dimension $V$ of the Diagonal Plane [P3]

> **Why $\rho = P(L > SLO)$ was removed:** It required the hallucinated Themis closed-form (Audit §A-1). $\Omega$ is its directly observable, paper-clean replacement.

### 5.3 Output Variables

| Variable | Symbol | Domain | Meaning |
|---|---|---|---|
| Vertical Scale Delta | $\Delta c$ | $\{-1, 0, +1, +2\}$ cores | CPU core adjustment per replica |
| Horizontal Scale Delta | $\Delta n$ | $\{-2, \ldots, +5\}$ replicas | Change in replica count |
| Scaling Mode | $M$ | {Vertical, Diagonal, Horizontal} | Primary action dimension |

### 5.4 Fuzzy Rule Base

Grounded in two verified paper insights:

1. **Themis [P1] economic insight:** prefer immediate in-place vertical adjustment (zero cold-start penalty) before spawning new replicas
2. **DiagonalScale [P3] efficiency insight:** diagonal moves on the $(H,V)$ plane frequently outperform pure horizontal or vertical moves

```
R1: IF Ψ is Moderate  AND Φ is Available
    → Mode=Vertical,    Δc=+1, Δn=0
    [Themis P1: prefer in-place before spawning replicas]

R2: IF Ψ is High      AND Φ is Available
    → Mode=Diagonal,    Δc=+1, Δn=⌈Ψ/2⌉
    [P3 §5.2: diagonal traversal beats pure H or V at high surge]

R3: IF Ψ is Critical  AND Φ is Exhausted
    → Mode=Horizontal,  Δc=0,  Δn=n_pred − n
    [Vertical ceiling hit; scale out — P1, P5]

R4: IF Ω is Tight     AND Φ is Available
    → Mode=Vertical,    Δc=+1, Δn=0
    [Fast latency fix, no cold-start overhead — P1]

R5: IF Ψ is Low       AND Ω is Ample
    → Mode=Diagonal,    Δc=−1, Δn=−1
    [Diagonal scale-down; P3 permits joint (H,V) reduction]

R6: IF Φ is Exhausted AND Ω is Comfortable
    → Mode=Horizontal,  Δc=0,  Δn=+2
    [Resources saturated, SLO still safe → gradual scale-out]
```

### 5.5 ANFIS Learning

Gaussian membership functions: $\mu(x) = e^{-(x - c_i)^2 / (2\sigma_i^2)}$

Training uses the ANFIS hybrid algorithm (Jang, 1993):
- **Consequent parameters:** closed-form least-squares
- **Antecedent parameters** ($c_i, \sigma_i$): gradient descent (backpropagation)

Training loss:

$$\mathcal{L}_{ANFIS} = \sum_t \Bigl[\alpha \cdot \bigl(L_{actual,t} - SLO\bigr)^2_+ + (1-\alpha) \cdot Cost_{actual,t}\Bigr]$$

where $(x)_+ = \max(0, x)$ penalizes only SLO violations, and $\alpha \in [0,1]$ is an operator-configurable SLO-vs-cost trade-off weight. Online fine-tuning runs every hour.

---

## 6. Layer 3 — NSGA-II Genetic Global Optimizer [P3 + NSGA-II]

### 6.1 The Diagonal Scaling Plane (from [P3])

Abdullah & Zaman (arXiv:2511.21612) define every resource configuration as a point $(H, V)$:
- $H \in \mathbb{Z}^+$: Horizontal replica count
- $V \in \mathbb{R}^d$: Per-replica resource vector (CPU cores, RAM)

The **Rebalance Penalty** for a transition $(H,V) \to (H',V')$, exactly as given in [P3]:

$$P_{rebalance}(H, V, H', V') = \lambda_1|H'-H| + \lambda_2\|V'-V\|_1 + \lambda_3 \cdot \text{ShardMovement}(H \to H')$$

where $\lambda_1, \lambda_2, \lambda_3$ are operator-configured penalty weights and ShardMovement estimates data migration cost when replica count changes.

### 6.2 Multi-Objective Formulation

Bi-objective Pareto problem:

$$\min_{(H', V')} \; \mathbf{F} = \bigl[f_1(H',V'),\;\; f_2(H',V')\bigr]$$

$$f_1 = H' \cdot P_{node} + \|V'\|_1 \cdot P_{resource} \quad\text{(infrastructure cost)}$$

$$f_2 = \mathbb{1}\!\left[L_{profile}(\hat{\lambda}, H', V') > SLO\right] \cdot \text{Penalty}_{SLO} \quad\text{(SLO violation indicator)}$$

Subject to: $H' \in \mathbb{Z}^+$, $V'_i \in [V_{min}, V_{max}]$

Transition overhead enters as an augmented cost:

$$f_1^{aug} = f_1 + \delta \cdot P_{rebalance}(H, V, H', V')$$

> **Latency estimation:** $L_{profile}(\hat{\lambda}, H', V')$ is computed from the offline profiling table built in the Themis style [P1] — not from any closed-form queuing equation.

### 6.3 Novel Chromosome Encoding — Scaling Trajectory

> **[Novel claim N3] — explicitly flagged as this proposal's primary contribution; not present in any reference paper.**

Each chromosome encodes a complete **scaling trajectory** over planning horizon $T$:

$$\mathbf{x} = \bigl[(H_1, V_1),\; (H_2, V_2),\; \ldots,\; (H_T, V_T)\bigr]$$

This allows NSGA-II to optimize the entire path on the Scaling Plane, penalizing trajectories with high cumulative rebalance overhead even when each individual step appears locally Pareto-optimal. P3's DIAGONALSCALE is a single-step greedy local search; this extends it to full trajectory optimization.

**Genetic operators:**
- **Crossover:** Single-point on time index $\{1,\ldots,T\}$; BLX-α blend on continuous $V$ components
- **Mutation:** $\pm 1$ integer perturbation on $H$; Gaussian additive noise on $V$
- **Selection:** Non-dominated sorting + crowding distance (standard NSGA-II, Deb et al. 2002)

### 6.4 GA–ANFIS Checkpoint Gating

> **[Novel claim N4]:** Slow global optimizer constraining a fast local decision-maker — not present in any reference paper.

NSGA-II runs every 15 minutes and outputs a Pareto-optimal checkpoint sequence $(H^*_\tau, V^*_\tau)_{\tau=1}^T$. After each ANFIS decision, if the proposed action would deviate by more than $\epsilon_H$ replicas or $\epsilon_V$ resource units from the nearest checkpoint, ANFIS output is clamped toward that checkpoint. This prevents high-frequency reactive decisions from accumulating into globally suboptimal drift.

---

## 7. Layer 0 — Multi-Level Monitoring (MAPE-K + AspectJ) [P4]

### 7.1 Three-Level Monitoring Hierarchy (from [P4])

| Level | Tool | Metrics Collected |
|---|---|---|
| Infrastructure | Prometheus / cAdvisor | CPU%, Memory%, Network I/O |
| Container | Kubernetes Metrics Server | Pod CPU requests, scheduling latency |
| **Application** | **AspectJ (load-time weaving)** | Internal queue depth, per-service response time, throughput |

The contribution of [P4] is that application-layer instrumentation surfaces demand signals before infrastructure-level saturation, giving the ANFIS engine earlier and higher-quality inputs. No specific timing numbers are claimed.

> **[Novel claim N5]:** P4 routes AspectJ signals into a threshold-based rule engine. This proposal routes them into the Prophet-LSTM forecaster's input feature space — coupling application-layer queue depth and latency with neural prediction. This combination does not appear in either P4 or P5.

### 7.2 Composite Stress Indicator (inspired by [P4])

$$\Sigma_{stress}(t) = w_1 \cdot \frac{CPU(t)}{CPU_{max}} + w_2 \cdot \frac{Q_{depth}(t)}{Q_{max}} + w_3 \cdot \frac{L_{app}(t)}{SLO}$$

where $w_1 + w_2 + w_3 = 1$. Weights are initialized uniformly and adjusted by the ANFIS feedback each cycle, prioritizing the metric that most reliably precedes SLO events.

---

## 8. Complete Algorithm (Verified)

```
NFG-DiagScale:

INPUT:  λ_hist (historical RPS), SLO target, cost budget B
OUTPUT: scaling actions (Δn, Δc, Mode) at each time step t

INITIALIZATION:
  Train Prophet-LSTM on λ_hist                              [P5]
  Build offline latency profile table: (b, c) → L_profile   [P1]
  Pre-train ANFIS on historical (state, action, outcome) logs
  Initialize NSGA-II: pop_size = 100, horizon T = 4 steps

MAPE-K LOOP (every Δt = 30 seconds):

  ── MONITOR ──
  1. Collect {CPU, Q_depth, L_app} via AspectJ + Prometheus  [P4]
  2. λ^KF_t ← KalmanFilter(λ^obs_t)                         [P2]
  3. Σ_stress ← w1·CPU/CPU_max + w2·Q/Q_max + w3·L_app/SLO  [P4]

  ── ANALYZE ──
  4. λ̂_{t+k} ← Prophet(t+k) + LSTM(residuals), k∈{1,5,15}m [P5]
  5. Ψ ← λ̂_{t+1} / λ^KF_t                                   [P5, P2]
  6. Ω ← (SLO − L_app) / SLO                                 [P1, P4]
  7. Φ ← 1 − c_curr / c_max                                   [P3]

  ── PLAN ──
  8. (M, Δc, Δn) ← ANFIS(Ψ, Ω, Φ)                          [ANFIS]
  9. IF t mod 15min = 0:
       (H*,V*)_{τ=1..T} ← NSGA-II(λ̂, L_profile, B, P_rebal) [P3, GA]
       Clamp ANFIS output to ε-neighbourhood of nearest checkpoint

  ── EXECUTE ──
  10. IF M = "Vertical"   → execute_vertical_scale(Δc)       [P1 in-place]
  11. IF M = "Diagonal"   → execute_vertical_scale(Δc)
                             schedule_horizontal_scale(Δn)    [P3]
  12. IF M = "Horizontal" → execute_horizontal_scale(Δn)      [P5]

  ── ONLINE LEARNING ──
  13. Append (Ψ, Ω, Φ, M, L_obs, Cost_obs) to rolling buffer
  14. Every 1h: retrain Prophet-LSTM on new data              [P5]
  15. Every 1h: ANFIS fine-tuning pass on rolling buffer
```

---

## 9. Objective Functions & KPIs

**Pareto objectives (NSGA-II):**

$$\min \; \mathbf{F} = \bigl[f_{cost},\;\; f_{SLO},\;\; f_{rebalance}\bigr]$$

**Scalarized loss (ANFIS training):**

$$\mathcal{L}_{scale} = \alpha \cdot f_{SLO} + (1-\alpha) \cdot f_{cost} + \mu_{rebal} \cdot f_{rebalance}$$

**Performance KPIs** (quantitative targets emerge from experiments; none pre-claimed):

| Metric | Formula |
|---|---|
| SLO Violation Rate | $SVR = \#\{t: L_t > SLO\} / T$ |
| Cost Efficiency Ratio | $CER = Cost_{baseline} / Cost_{NFG}$ |
| Prediction MAPE | $\frac{1}{T}\sum_t \lvert\lambda_t - \hat{\lambda}_t\rvert / \lambda_t$ |
| Rebalance Overhead | $\sum_t P_{rebalance}(t)$ |
| Scaling Reaction Lag | $\tau_{react} = t_{scale} - t_{detect}$ |

---

## 10. Novelty Claims (Tightened and Honest)

| ID | Claim | Reference Basis | Gap Filled |
|---|---|---|---|
| **N1** | Multi-timescale fusion: Prophet-LSTM (long-horizon planning) + Kalman Filter (intra-window noise suppression) in one pipeline | P5 (Prophet-LSTM), P2 (Kalman for RPS) | P5 has no noise-suppression; P2 has no planning horizon; neither combines them |
| **N2** | ANFIS as decision engine on the Diagonal Scaling Plane (H,V) | P3 (Scaling Plane), P1 (economic insight), P4 (MAPE-K), ANFIS classical | P3 uses greedy local search; P4 uses threshold rules; no prior work uses ANFIS on (H,V) plane |
| **N3** | NSGA-II encoding T-step scaling trajectories as chromosomes | P3 (Scaling Plane + Rebalance Penalty), NSGA-II classical | P3's DIAGONALSCALE is single-step; trajectory chromosome encoding is novel |
| **N4** | GA checkpoint gating of ANFIS (slow global optimizer constrains fast local decisions) | None — fully original integration | Not present in any reference paper |
| **N5** | AspectJ application-layer queue/latency signals routed into neural forecaster feature space | P4 (AspectJ probes), P5 (Prophet-LSTM) | P4 feeds a rule engine; P5 uses HTTP logs; neither routes AspectJ signals into NN inputs |

---

## 11. Baselines & Experimental Plan

### Baselines

| System | Type |
|---|---|
| Kubernetes HPA (CPU threshold) | Reactive, Horizontal only |
| Kubernetes VPA | Reactive, Vertical only |
| Themis [P1] | Hybrid H+V (profiling + DP/IP) |
| Prophet-LSTM HPA [P5] | Proactive, Horizontal only |
| DiagonalScale [P3] | Unified H+V (greedy local search) |
| **NFG-DiagScale (Ours)** | **Proactive + ANFIS + GA trajectory + Multi-timescale fusion** |

### Datasets

- **NASA HTTP Access Log (1998):** Diurnal and seasonal patterns — used and validated in P5
- **FIFA World Cup '98 Log:** Extreme burst traffic — used and validated in P5
- **Azure Public Dataset v2 (2019):** Multi-service microservice traces with resource telemetry

### Environment

- **Platform:** Kubernetes (KIND or minikube multi-node)
- **Load Generator:** Locust (HTTP) or synthetic trace replay
- **Workload Patterns:** Step, Ramp, Spike, Diurnal, Mixed

---

## 12. Final AI Component Summary

| AI Technique | Component | Role | Basis |
|---|---|---|---|
| **Neural Network** | Prophet-LSTM Forecaster | Multi-timescale workload prediction | P5 — Guruge & Priyadarshana (2025) |
| **Neural Network** | ANFIS backprop/least-squares layer | Trains Gaussian MF parameters from historical scaling outcomes | ANFIS — Jang (1993) |
| **Fuzzy Logic** | ANFIS Mamdani rule base | Interpretable, adaptive mode and magnitude decisions | P1, P3, P4 insights encoded as rules |
| **Genetic Algorithm** | NSGA-II Trajectory Optimizer | Pareto-optimal T-step path on Diagonal Scaling Plane | P3 (plane + penalty) + NSGA-II — Deb (2002) |

> **Signal Processing (not AI):** Kalman Filter (Bayesian state estimator) for intra-window RPS noise suppression — sourced from P2.

---

## 13. References

1. Razavi, K., Salmani, M., Mühlhäuser, M., Koldehofe, B., Wang, L. (2024). *A Tale of Two Scales: Reconciling Horizontal and Vertical Scaling for Inference Serving Systems* (Themis). arXiv:2407.14843.

2. Gu, J., Wang, P., Araya, I.D.N., Huang, K., Gerndt, M. (2025). *HAS-GPU: Efficient Hybrid Auto-scaling with Fine-grained GPU Allocation for SLO-aware Serverless Inferences*. arXiv:2505.01968.

3. Abdullah, S., Zaman, S.R. (2025). *Diagonal Scaling: A Multi-Dimensional Resource Model and Optimization Framework for Distributed Databases*. arXiv:2511.21612.

4. Solino, A., Batista, T., Cavalcante, E. (2025). *An Autonomic Computing Approach for Scaling Cloud-Based Smart City Platforms*. ACM UCC'25, DOI: 10.1145/3773274.3774256.

5. Guruge, P.B., Priyadarshana, Y.H.P.P. (2025). *Time Series Forecasting-based Kubernetes Autoscaling using Facebook Prophet and Long Short-Term Memory*. Frontiers in Computer Science, Vol.7, doi:10.3389/fcomp.2025.1509165.

6. Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. *IEEE Trans. Evol. Comput.*, 6(2), 182–197.

7. Jang, J.-S.R. (1993). ANFIS: Adaptive-Network-Based Fuzzy Inference System. *IEEE Trans. Syst. Man Cybern.*, 23(3), 665–685.
