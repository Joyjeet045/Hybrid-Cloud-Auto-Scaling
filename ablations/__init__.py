"""Negative-result ablation studies for NF-DiagScale.

Each subpackage is a self-contained, *pluggable* experiment that plugs into the
unmodified core controller/engine via subclassing, so the baseline architecture
and all reported figures are preserved exactly. None of these are enabled on the
default path.

- ``rec1_gnn_selection`` -- PDF Rec 2 / Approach 1: replace the analytic
  critical-path selector with a distilled GCN (strong negative result).
- ``rec2_adaptive_mf``   -- PDF Rec 2 / Approach 2: replace the expert membership
  functions with data-driven (clustered) ones (neutral-to-negative result).
"""
