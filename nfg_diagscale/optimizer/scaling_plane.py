"""
Diagonal Scaling Plane surface models.

[P3] Abdullah & Zaman (2025), "Diagonal Scaling", arXiv:2511.21612

  Section III-B, Node-Intrinsic Latency:
    L_node(V) = alpha/c + beta/r + gamma/b + delta/s

  Where V = (c, r, b, s): CPU cores, RAM GB, bandwidth Gbps, storage IOPS
  alpha, beta, gamma, delta are sensitivity constants.

  Section III-C, Coordination Latency:
    L_coord(H) = eta * log(H) + mu * H^theta

  Where 0 < theta < 1 creates super-logarithmic but sub-linear growth,
  matching empirical measurements in distributed consensus literature.

  Section III-D, Total Latency:
    L(H, V) = L_node(V) + L_coord(H)

  Section III-E, Throughput Surface:
    T_node(V) = kappa * min(c, r, b, s)
    T(H, V)   = H * T_node(V) * phi(H)
    phi(H)    = 1 / (1 + omega * log(H))

  Section III-G, Monetary Cost Surface:
    C(H, V) = H * C_node(V)

  Section III-H, Multi-Objective Optimization:
    F(H, V) = alpha_L * L(H,V) + beta_C * C(H,V) + gamma_K * K(H,V)
    subject to: L(H,V) <= L_max, T(H,V) >= T_min

  Section IV-C, Lemma 1:
    "If dF/dH != 0 and dF/d||V|| != 0, then the optimal direction
     is neither horizontal nor vertical."
"""
import numpy as np


class ScalingPlane:
    def __init__(self, config):
        sp = config["scaling_plane"]
        # [P3 sect III-B] Node latency sensitivity constants
        self.alpha = sp["alpha"]
        self.beta = sp["beta"]
        self.gamma_sp = sp["gamma_sp"]
        self.delta_sp = sp["delta_sp"]

        # [P3 sect III-C] Coordination latency parameters
        self.eta_coord = sp["eta_coord"]
        self.mu_coord = sp["mu_coord"]
        self.theta = sp["theta"]

        # [P3 sect III-G] Cost parameters
        self.cost_per_core = sp["cost_per_core"]
        self.cost_per_gb_ram = sp["cost_per_gb_ram"]
        self.cost_per_replica = sp["cost_per_replica"]

    def node_latency(self, c, r, b, s):
        """
        [P3 sect III-B] L_node(V) = alpha/c + beta/r + gamma/b + delta/s
        """
        c = max(c, 0.5)
        r = max(r, 0.1)
        b = max(b, 0.01)
        s = max(s, 1.0)
        return self.alpha / c + self.beta / r + self.gamma_sp / b + self.delta_sp / s

    def coordination_latency(self, H):
        """
        [P3 sect III-C] L_coord(H) = eta * log(H) + mu * H^theta
        """
        H = max(H, 1)
        return self.eta_coord * np.log(H) + self.mu_coord * (H ** self.theta)

    def total_latency(self, H, c, r, b, s):
        """
        [P3 sect III-D] L(H, V) = L_node(V) + L_coord(H)
        """
        return self.node_latency(c, r, b, s) + self.coordination_latency(H)

    def node_cost(self, c, r):
        """
        [P3 sect III-G] C_node(V) based on resource pricing.
        """
        return self.cost_per_core * c + self.cost_per_gb_ram * r

    def total_cost(self, H, c, r):
        """
        [P3 sect III-G] C(H, V) = H * C_node(V)
        """
        return H * self.node_cost(c, r) + H * self.cost_per_replica

    def objective(self, H, c, r, b, s, slo, alpha_w=0.4, beta_w=0.4, gamma_w=0.2):
        """
        [P3 sect III-H] F(H, V) = alpha*L + beta*C + gamma*K
        Scalarized multi-objective function.
        """
        lat = self.total_latency(H, c, r, b, s)
        cost = self.total_cost(H, c, r)

        # Normalize latency by SLO
        lat_norm = lat / slo
        cost_norm = cost

        return alpha_w * lat_norm + beta_w * cost_norm + gamma_w * lat_norm * cost_norm

    def is_feasible(self, H, c, r, b, s, slo, min_throughput=0):
        """
        [P3 sect III-H] Feasibility constraints:
          L(H,V) <= L_max  and  T(H,V) >= T_min
        """
        lat = self.total_latency(H, c, r, b, s)
        return lat <= slo

    def gradient_direction(self, H, c, r, b, s, slo, delta_h=1, delta_c=1):
        """
        [P3 sect IV-B] Gradient alignment condition.
        Compute numerical gradient of F to determine optimal scaling direction.
        """
        F_curr = self.objective(H, c, r, b, s, slo)
        F_dh = self.objective(H + delta_h, c, r, b, s, slo)
        F_dc = self.objective(H, c + delta_c, r, b, s, slo)

        dF_dH = F_dh - F_curr
        dF_dV = F_dc - F_curr

        # [P3 Lemma 1] If both partial derivatives non-zero, diagonal is optimal
        if abs(dF_dH) > 1e-8 and abs(dF_dV) > 1e-8:
            return "diagonal", dF_dH, dF_dV
        elif abs(dF_dH) > 1e-8:
            return "horizontal", dF_dH, dF_dV
        elif abs(dF_dV) > 1e-8:
            return "vertical", dF_dH, dF_dV
        else:
            return "stable", dF_dH, dF_dV
