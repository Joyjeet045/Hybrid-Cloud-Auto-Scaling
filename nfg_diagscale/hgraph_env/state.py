"""Torch-free state extraction for the vendored HGraphScale simulator.

The original HGraphScale policy consumes a PyTorch-Geometric hierarchical graph
(PM -> VM -> container) built by ``utils.utils.graph_construct``. NF-DiagScale
does not use a GNN, so we read the same underlying quantities directly from the
``cloud_simulator`` object into a plain, numpy-friendly :class:`CloudState`.

Every feature below is a quantity the simulator already maintains; we do not add
new physics. The mapping to the papers:

* ``vcpu`` / ``max_scal_vcpu``      -> container resource & vertical headroom.
* ``qlen`` / ``pending_time``       -> queue backlog (M/M/1 occupancy proxy).
* ``aver_resptime``                 -> per-container mean response time (ms).
* ``workload_his``                  -> per-slot request counts (temporal signal,
                                       STAR "temporal workload variations").
* ``rank``                          -> upward rank on the application DAG
                                       (HEFT, Topcuoglu et al. 2002): captures the
                                       "spatial dependency" that STAR encodes with
                                       a GNN, but as an explicit critical-path score.
* ``total_cost`` / ``budget``       -> cost side of the objective (Eq. 9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContainerState:
    """Snapshot of a single container (microservice replica)."""

    con_id: int
    con_type: int
    vcpu: float
    max_scal_vcpu: float
    qlen: int
    pending_time: float
    aver_resptime: float
    request_num: int
    workload_his: np.ndarray
    rank: float


@dataclass
class CloudState:
    """Full torch-free observation passed to the NF-DiagScale controller."""

    containers: list[ContainerState]
    map_type_to_conids: dict[int, list[int]]
    num_microservices: int
    total_cost: float
    budget: float
    deadline: float
    slot_index: int
    num_vms: int = 0
    rank: dict[int, float] = field(default_factory=dict)
    proc_time: dict[int, float] = field(default_factory=dict)

    @property
    def cost_headroom(self) -> float:
        """Remaining budget fraction in [.,1]; <0 means over budget."""
        if self.budget <= 0:
            return 0.0
        return (self.budget - self.total_cost) / self.budget


def _compute_upward_rank(dag, proc_attr: str = "processTime") -> dict[int, float]:
    """HEFT upward rank on the static application DAG (Topcuoglu et al., 2002).

    rank(n) = proc(n) + max_{m in succ(n)} ( comm(n, m) + rank(m) ),
    with rank(exit) = proc(exit). Higher rank => more downstream work depends on
    the node => more critical to scale. Communication weights use the DAG edge
    ``weight`` attribute when present (set by buildDAGfromXML).
    """
    rank: dict[int, float] = {}

    def _proc(n) -> float:
        return float(dag.nodes[n].get(proc_attr, 0.0))

    try:
        import networkx as nx

        order = list(nx.topological_sort(dag))
    except Exception:
        order = list(dag.nodes())
    for n in reversed(order):
        succ = list(dag.successors(n)) if hasattr(dag, "successors") else []
        if not succ:
            rank[n] = _proc(n)
        else:
            best = 0.0
            for m in succ:
                comm = 0.0
                if dag.has_edge(n, m):
                    comm = float(dag.edges[n, m].get("weight", 0.0))
                best = max(best, comm + rank.get(m, 0.0))
            rank[n] = _proc(n) + best
    return rank


def get_static_rank(sim) -> dict[int, float]:
    """Upward rank per microservice id, computed once and cached on ``sim``."""
    cached = getattr(sim, "_nfg_static_rank", None)
    if cached is not None:
        return cached
    dag = sim.set.dataset.wset[0]
    rank = _compute_upward_rank(dag)
    if rank:
        max_r = max(rank.values()) or 1.0
        rank = {k: v / max_r for k, v in rank.items()}
    sim._nfg_static_rank = rank
    return rank


def get_static_proc_time(sim) -> dict[int, float]:
    """Base processing time (``processTime``, ms) per microservice id, cached.

    This is the simulator's own per-task execution time on one vCPU
    (``runtime / scale``; HGraphScale Eq. 1), read straight from the DAG so the
    queue model sees identical service times to the simulator.
    """
    cached = getattr(sim, "_nfg_static_proc", None)
    if cached is not None:
        return cached
    dag = sim.set.dataset.wset[0]
    proc = {n: float(dag.nodes[n].get("processTime", 0.0)) for n in dag.nodes()}
    sim._nfg_static_proc = proc
    return proc


def extract_state(sim) -> CloudState:
    """Build a :class:`CloudState` from a live ``cloud_simulator`` instance."""
    rank = get_static_rank(sim)
    proc_time = get_static_proc_time(sim)

    containers: list[ContainerState] = []
    for con_id, con in sim.con_queues.items():
        if not getattr(con, "active", True):
            continue
        con_type = con.get_contype()
        containers.append(
            ContainerState(
                con_id=con_id,
                con_type=con_type,
                vcpu=float(con.get_vcpu()),
                max_scal_vcpu=float(con.get_max_scal_vcpu()),
                qlen=int(con.conQueue.qlen()),
                pending_time=float(getattr(con, "pendingTaskTime", 0.0)),
                aver_resptime=float(getattr(con, "aver_resptime", 0.0)),
                request_num=int(getattr(con, "request_num", 0)),
                workload_his=np.asarray(getattr(con, "workload_his", np.array([])), dtype=float),
                rank=float(rank.get(con_type, 0.0)),
            )
        )

    slot_index = int(len(getattr(sim, "step_cost", [])))
    num_vms = int(len(getattr(sim, "vm_queues", [])))

    return CloudState(
        containers=containers,
        map_type_to_conids={k: list(v) for k, v in sim.map_con_type_id.items()},
        num_microservices=int(sim.num_app),
        total_cost=float(sim.total_cost),
        budget=float(sim.budget),
        deadline=float(getattr(sim, "deadline", 500.0)),
        slot_index=slot_index,
        num_vms=num_vms,
        rank=rank,
        proc_time=proc_time,
    )
