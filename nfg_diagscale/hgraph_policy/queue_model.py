"""Grounded latency and cost models for the HGraphScale environment.

These replace NF-DiagScale's original Themis latency look-up table with closed-
form models whose physics match the vendored simulator, so the optimizer and the
ANFIS engine reason about the *same* dynamics the simulator actually executes.

Latency (M/D/1 transient batch drain).  In the HGraphScale simulator each control
interval delivers a burst of requests; a container of microservice ``i`` runs its
tasks one at a time with deterministic service time ``D = et_i / vcpu`` ms (this
is the simulator's own rule ``ET = et / con_cpu``; HGraphScale Eq. 1). For a batch
of ``q`` tasks served FIFO on a single server, task ``k`` completes at ``k*D``, so
the mean response time over the batch is

    R = D * (1 + (q - 1) / 2),                                   (queue drain)

i.e. service time plus mean queueing delay ``(q-1)/2 * D``. This is the standard
transient/batch M/D/1 result (Kleinrock, 1975, *Queueing Systems, Vol. 1*). With
``n`` replicas under CWRR load balancing (HGraphScale Eq. 15) each replica sees
``q = lambda / n`` tasks.

Cost (VM rental).  HGraphScale Eq. 5 charges each active VM ``price * hours``. The
marginal cost of a scaling action is therefore the price of any *additional* VMs
its extra vCPUs force to be rented, prorated over the remaining horizon.
"""
from __future__ import annotations

import math


def batch_response_time(lam: float, et: float, vcpu: float, replicas: int) -> float:
    """Mean per-container response time (ms) for a burst of ``lam`` tasks.

    ``et`` is the microservice base processing time (DAG ``processTime``); the
    simulator's deterministic service time is ``D = et / vcpu`` (HGraphScale
    Eq. 1). Uses the M/D/1 batch-drain mean (Kleinrock, 1975).
    """
    vcpu = max(vcpu, 1e-6)
    replicas = max(int(replicas), 1)
    D = et / vcpu
    q = lam / replicas
    queue_delay = max(0.0, (q - 1.0) / 2.0) * D
    return D + queue_delay


def load_factor_cwrr(lam: float, et: float, total_vcpu: float, deadline: float) -> float:
    """CWRR-weighted load/pressure ``psi`` for a (possibly heterogeneous) replica set.

    HGraphScale balances a microservice's burst across its replicas with
    Cost-Weighted Round Robin (Eq. 15): replica ``j`` receives a share of the load
    proportional to its vCPU, ``W_j = c_j / sum_k c_k``. Under CWRR every replica's
    drain time collapses to the *same* value regardless of how vCPU is distributed:

        q_j * D_j = (lam * c_j / sum c) * (et / c_j) = lam * et / sum_k c_k,

    so the bottleneck pressure depends only on the microservice's *total* vCPU.
    This matches the simulator's load balancer exactly and is robust to the
    heterogeneous replicas that diagonal scaling produces (one replica boosted
    first) without any mean-vCPU rounding.
    """
    total_vcpu = max(float(total_vcpu), 1e-6)
    drain_time = lam * et / total_vcpu
    return drain_time / max(deadline, 1e-6)


def marginal_vm_cost(old_total_vcpu: float, new_total_vcpu: float,
                     vm_size: float, vm_price_per_hr: float,
                     hours_remaining: float) -> float:
    """Cost (USD) of the extra VMs required to grow allocation (HGraphScale Eq. 5).

    VMs are sized ``vm_size`` vCPU at ``vm_price_per_hr``; the number of VMs needed
    is ``ceil(total_vcpu / vm_size)``. Only *additional* VMs incur new cost; the
    charge is prorated over ``hours_remaining`` of the evaluation horizon.
    """
    vm_size = max(vm_size, 1e-6)
    vms_old = math.ceil(max(old_total_vcpu, 0.0) / vm_size)
    vms_new = math.ceil(max(new_total_vcpu, 0.0) / vm_size)
    delta_vms = max(0, vms_new - vms_old)
    return delta_vms * vm_price_per_hr * max(hours_remaining, 0.0)
