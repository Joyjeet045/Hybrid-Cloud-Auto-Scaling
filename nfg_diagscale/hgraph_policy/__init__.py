"""NFG-DiagScale controller migrated to the HGraphScale heterogeneous-container env.

This package re-homes the NFG-DiagScale brain (Kalman forecasting + ANFIS
neuro-fuzzy decision + a genetic/multi-objective magnitude optimizer) onto the
vendored HGraphScale discrete-event simulator, replacing the original single-
service toy environment, the Themis latency look-up, and the Prophet-LSTM stack.

Modules:
* :mod:`forecaster`  - per-container temporal load prediction (Kalman + Holt).
* :mod:`queue_model` - grounded M/D/1 batch-drain latency model + VM cost model.
* :mod:`optimizer`   - NSGA-II magnitude optimizer over (replicas, vCPU).
* :mod:`controller`  - the closed-loop policy mapping ``CloudState`` -> action.
"""
