"""Vendored HGraphScale heterogeneous-container simulator (torch/gym-free).

This package vendors the discrete-event cloud simulator published with
HGraphScale (Fang et al., IEEE TSC 2026, arXiv:2511.01881) and reused by the
STAR paper (Fang et al., Expert Systems With Applications 2026). The original
code is at https://github.com/sine-fandel/HGraphScale .

Only the simulator core (``env/autoscaling_v1``) is vendored. The original
PyTorch-Geometric state builder (``utils.utils.graph_construct``) and the gym
dependency are intentionally replaced by lightweight shims so that the
simulator runs on a minimal ``numpy + networkx`` stack. Our NFG-DiagScale
policy reads the simulator state directly (see :mod:`nfg_diagscale.hgraph_env.state`)
instead of using the GNN graph tensors.

Importing this package prepends the vendored directory to ``sys.path`` so the
internal absolute imports (``from env.autoscaling_v1...``, ``from config.param
import configs``, ``from utils.utils import graph_construct``) resolve to the
vendored copies.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
