"""Torch-free configuration shim for the vendored HGraphScale simulator.

The original ``config/param.py`` builds an ``argparse`` namespace and parses
``sys.argv`` at import time, which is unsafe when the simulator is imported as a
library. This shim exposes the same ``configs`` object with the defaults used in
the STAR / HGraphScale experiments, without touching ``sys.argv``.

Grounding of the values (all from the papers):
- ``scale``    : runtime-to-processTime divisor in buildDAGfromXML (HGraphScale code default 2).
- ``penalty``  : budget-violation penalty coefficient. STAR sec. 5.2 sets phi=100
                 ("following Shi et al., 2023"); HGraphScale uses rho=100 (Eq. 9).
- ``budget``   : 200 USD/day (STAR sec. 5.2; HGraphScale Eq. 8).
- ``time_slot``: 180 s = 3-min scaling interval (STAR sec. 5.1).
"""
from types import SimpleNamespace

configs = SimpleNamespace(
    scale=2,
    penalty=100,
    budget=200,
    time_slot=180,
    vm_types=5,
    seed=0,
    app_num=1,
    app_size="A11",
    workload_pattern=0,
)
