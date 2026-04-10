"""
Rebalance penalty for scaling transitions.
"""
import numpy as np


class RebalancePenalty:
    def __init__(self, config):
        rcfg = config["rebalance"]
        # Penalty weights
        self.lambda1 = rcfg["lambda1"]
        self.lambda2 = rcfg["lambda2"]
        self.lambda3 = rcfg["lambda3"]

    def compute(self, H_curr, V_curr, H_new, V_new):
        """
        Computes rebalance penalty based on configuration changes.
        V_curr and V_new are tuples/arrays of (c, r, b, s).
        """
        # Horizontal transition cost
        h_delta = abs(H_new - H_curr)

        # Vertical transition cost (L1 norm)
        v_curr = np.array(V_curr, dtype=float)
        v_new = np.array(V_new, dtype=float)
        v_delta = np.sum(np.abs(v_new - v_curr))

        # Shard movement cost proportional to replica count change
        shard_movement = self._estimate_shard_movement(H_curr, H_new)

        penalty = (self.lambda1 * h_delta
                   + self.lambda2 * v_delta
                   + self.lambda3 * shard_movement)
        return penalty

    def _estimate_shard_movement(self, H_old, H_new):
        """
        Consistent hashing model: when going from H to H' nodes,
        approximately (1 - H/H') fraction of shards must move for expansion,
        or (1 - H'/H) for contraction.
        """
        if H_old == H_new:
            return 0.0
        if H_new > H_old:
            return H_old * (1.0 - H_old / H_new)
        else:
            return H_new * (1.0 - H_new / H_old)
