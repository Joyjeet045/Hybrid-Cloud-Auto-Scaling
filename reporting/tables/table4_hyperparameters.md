**NF-DiagScale configuration (default.yaml)**

| Component | Parameter | Value |
| --- | --- | --- |
| Kalman filter | A | 1.0 |
| Kalman filter | H | 1.0 |
| Kalman filter | Q | 0.01 |
| Kalman filter | D | 0.1 |
| Kalman filter | initial_P | 1.0 |
| Holt forecast | holt_alpha | 0.5 |
| Holt forecast | holt_beta | 0.3 |
| ANFIS deadzones | deadzone_low | 0.25 |
| ANFIS deadzones | deadzone_moderate | 0.4 |
| ANFIS deadzones | deadzone_near_capacity | 0.55 |
| ANFIS deadzones | deadzone_over_capacity | 0.15 |
| Online learning | enabled | True |
| Online learning | eta | 0.1 |
| Online learning | kappa_slo | 1.0 |
| Online learning | kappa_cost | 0.5 |
| Online learning | beta | 0.9 |
| Online learning | deadzone_eps | 0.05 |
| Online learning | budget_pacing | True |
| Online learning | bound_dc | 6.0 |
| Online learning | bound_dn | 5.0 |
| Sizer rebalance | penalty_multiplier | 1.5 |
| Cloud bounds | min_replicas | 1 |
| Cloud bounds | max_replicas | 15 |
| Cloud bounds | min_cores | 1 |
| Cloud bounds | max_cores | 16 |
| Cloud bounds | pod_max_rps | 20 |
| Cloud bounds | budget | 200.0 |
| Controller | idle_pressure | 0.25 |
| Controller | budget_safety | 0.97 |
| Controller | max_res | 4 |
| SLO | slo_ms | 500.0 |
