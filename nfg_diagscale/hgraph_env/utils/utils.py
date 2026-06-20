"""Torch-free shim of ``utils.utils`` for the vendored simulator.

The original ``utils.utils.graph_construct`` builds PyTorch-Geometric tensors for
the CHGNN / GAT policy of HGraphScale. NFG-DiagScale does not use a GNN; it reads
the cloud state directly (see :mod:`nfg_diagscale.hgraph_env.state`). We therefore
provide a stub so ``from utils.utils import graph_construct`` succeeds without
importing ``torch`` / ``torch_geometric``.

The stub is never reached on our policy path: we subclass the base
``cloud_simulator`` whose ``reset()`` / ``step()`` never call ``graph_construct``
(only the original ``ASEnv`` wrapper did).
"""


def graph_construct(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError(
        "graph_construct (PyG/GNN state) is disabled in the torch-free vendored "
        "env. Use nfg_diagscale.hgraph_env.state.extract_state instead."
    )
