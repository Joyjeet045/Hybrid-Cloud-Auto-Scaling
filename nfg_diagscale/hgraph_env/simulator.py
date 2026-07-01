"""Thin, torch-free wrapper around the vendored HGraphScale ``cloud_simulator``.

We subclass the *base* simulator (not ``ASEnv``) so we never touch the
PyTorch-Geometric ``graph_construct`` path. The control loop mirrors the
original exactly:

    reward, done, response_time, total_cost = super().step(self.nextTimeStep, action)

with the HGraphScale action contract ``action = (selected_con, scaling,
inverse_new_id_map)`` (see ``cloud_simulator.hges_auto_scaling``):
  * ``selected_con``       : container index that ``inverse_new_id_map`` maps to a
                             live container id;
  * ``scaling``            : signed integer vCPU delta (>0 scale-up/out,
                             <0 scale-in, 0 no-op);
  * ``inverse_new_id_map`` : index -> live-container-id map. Because our policy
                             already works with live ids, we pass the identity
                             map over ``self.con_queues``.

The reward is HGraphScale Eq. 9:
    r = -max(0, penalty * (total_cost - budget)) - mean(response_time).
"""
from __future__ import annotations

import nfg_diagscale.hgraph_env

from env.autoscaling_v1.lib.cloud_env_maxPktNum import cloud_simulator

from nfg_diagscale.hgraph_env.state import CloudState, extract_state

WORKLOAD_PATTERNS = {
    "nasa": 0,
    "wiki": 1,
    "alibaba": 3,
}


class HGraphScaleEnv(cloud_simulator):
    """Heterogeneous-container autoscaling environment (HGraphScale simulator)."""

    def __init__(
        self,
        app: str,
        workload: str | int,
        seed: int = 0,
        budget: float = 200.0,
        app_num: int = 1,
    ):
        if isinstance(workload, str):
            pattern = WORKLOAD_PATTERNS[workload.lower()]
        else:
            pattern = int(workload)

        args = {
            "seed": seed,
            "envid": 0,
            "app_size": app,
            "app_num": app_num,
            "app_types": app,
            "workload_pattern": pattern,
            "budget": budget,
        }
        super().__init__(args)
        self._seed = seed
        self.app = app
        self.workload = workload

    def reset(self, test: bool = True) -> CloudState:
        """Reset the simulator and return the initial torch-free state."""
        super().reset(self._seed, test=test)
        return extract_state(self)

    def step(self, decision):
        """Apply one scaling decision and simulate one 3-min control interval.

        ``decision`` is either ``None``/empty (no-op) or a ``(con_id, vcpu_delta)``
        tuple selecting a single live container to scale, matching HGraphScale's
        one-action-per-interval design.

        Returns ``(state, reward, done, info)`` where ``info`` is the simulator's
        ``episode_info`` dict once the episode ends (empty otherwise).
        """
        action = self._build_action(decision)
        reward, done, _response_time, _total_cost = super().step(self.nextTimeStep, action)
        state = extract_state(self)
        info = getattr(self, "episode_info", {}) if done else {}
        return state, reward, done, info

    def _build_action(self, decision):
        """Translate a CDA command batch into the HGraphScale batch action."""
        inverse_map = {cid: cid for cid in self.con_queues}
        if not decision:
            return ([], inverse_map)
        return (list(decision), inverse_map)
