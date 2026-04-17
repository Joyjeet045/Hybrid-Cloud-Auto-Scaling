"""
Diagonal Scaling Plane surface models.
"""
import numpy as np


class ScalingPlane:
    def __init__(self, config):
        sp = config["scaling_plane"]
        # Cost parameters
        self.cost_per_core = sp["cost_per_core"]
        self.cost_per_gb_ram = sp["cost_per_gb_ram"]
        self.cost_per_replica = sp["cost_per_replica"]

        # Use pod_max_rps from cloud config for consistency
        self.pod_max_rps = config["cloud"]["pod_max_rps"]

    def node_cost(self, c, r):
        """
        Computes node cost based on resource pricing.
        """
        return self.cost_per_core * c + self.cost_per_gb_ram * r

    def total_cost(self, H, c, r):
        """
        Computes total monetary cost.
        """
        return H * self.node_cost(c, r) + H * self.cost_per_replica
