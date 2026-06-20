"""Adaptive Neuro-Fuzzy Inference System (ANFIS) decision engine.

The "Neuro-Fuzzy" component of NFG-DiagScale. Implements a first-order
Takagi-Sugeno fuzzy inference system (Takagi & Sugeno, 1985) arranged as the
five-layer ANFIS architecture of Jang (1993): Gaussian membership functions
(layer 1), product T-norm rule firing (layer 2), normalisation (layer 3),
linear consequents ``f_i = p_i*psi + q_i*omega + r_i*phi + s_i`` (layer 4), and
the weighted-average defuzzified output (layer 5). The engine runs in inference
mode (consequent parameters held fixed), so it reduces to a stable rule-weighted
controller whose output biases the diagonal (replicas, vCPU) scaling decision.
"""
import numpy as np
from nfg_diagscale.decision.fuzzy_rules import (
    build_rule_base, gaussian_mf, LINGUISTIC_TERMS, MODE_NAMES,
    SCALING_MODES
)


class ANFISEngine:
    def __init__(self, config):
        self.config = config
        acfg = config["anfis"]
        # SLO deadline (ms). Inference-only engine: no online learning, so the
        # legacy learning-rate / cost-model keys are no longer read.
        self.slo = config.get("slo_ms", 500)
        self.pod_max_rps = config["cloud"]["pod_max_rps"]
        self.max_cores = config["cloud"]["max_cores"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.min_cores = config["cloud"]["min_cores"]

        # GA checkpoint influence weight (1 - this = ANFIS weight)
        self._ga_influence = acfg.get("ga_influence_weight", 0.3)

        # Adaptive deadzone widths by load regime
        self._dz_low = acfg.get("deadzone_low", 0.2)
        self._dz_moderate = acfg.get("deadzone_moderate", 0.35)
        self._dz_near_cap = acfg.get("deadzone_near_capacity", 0.5)
        self._dz_over_cap = acfg.get("deadzone_over_capacity", 0.15)

        self.rules = build_rule_base()

        # Premise parameters: centers and sigmas for each MF
        self.mf_params = {}
        for var_name, terms in LINGUISTIC_TERMS.items():
            self.mf_params[var_name] = {}
            for term_name, (center, sigma) in terms.items():
                self.mf_params[var_name][term_name] = {
                    "center": center,
                    "sigma": sigma,
                }

        # Consequent parameters for Takagi-Sugeno output
        # Each rule has linear consequent: f_i = p_i*psi + q_i*omega + r_i*phi + s_i
        self.consequent_params = {}
        for rule in self.rules:
            self.consequent_params[rule.rule_id] = {
                "p": 0.0,
                "q": 0.0,
                "r": 0.0,
                "s_mode": float(rule.mode),
                "s_dc": float(rule.delta_c),
                "s_dn": float(rule.delta_n),
            }

    def _compute_memberships(self, inputs):
        """Layer 1: Fuzzification using Gaussian MFs."""
        memberships = {}
        for var_name, value in inputs.items():
            if var_name not in self.mf_params:
                continue
            memberships[var_name] = {}
            for term_name, params in self.mf_params[var_name].items():
                mu = gaussian_mf(value, params["center"], params["sigma"], term_name)
                memberships[var_name][term_name] = mu
        return memberships

    def _compute_firing_strengths(self, inputs):
        """
        Layer 2: Rule firing strengths w_i = product of membership values.
        """
        strengths = []
        for rule in self.rules:
            w = 1.0
            for var_name, term_name in rule.antecedents.items():
                if var_name in inputs:
                    params = self.mf_params[var_name][term_name]
                    mu = gaussian_mf(inputs[var_name], params["center"], params["sigma"], term_name)
                    w *= mu
            strengths.append(w)
        return np.array(strengths)

    def _normalize_strengths(self, strengths):
        """
        Layer 3: Normalized firing strengths.
        w_bar_i = w_i / sum(w_k)
        """
        total = np.sum(strengths)
        if total < 1e-12:
            return np.ones(len(strengths)) / len(strengths)
        return strengths / total

    def _compute_consequents(self, inputs, n_current, cores_current, predicted_rps):
        """Layer 4: Consequent calculation (Adaptive Takagi-Sugeno)."""
        consequent_outputs = []
        for rule in self.rules:
            cp = self.consequent_params[rule.rule_id]

            psi = inputs.get("psi", 1.0)
            omega = inputs.get("omega", 0.5)
            phi = inputs.get("phi", 0.5)

            # True Takagi-Sugeno First Order Consequents:
            # f_i = s_i + p_i*psi + q_i*omega + r_i*phi
            mode_raw = cp["s_mode"] + cp["p"] * psi + cp["q"] * omega
            
            # The learned adjustments (p, q, r) steer BOTH vertical and horizontal decisions 
            # away from the base rule constants to fix prediction errors
            dc_raw = cp["s_dc"] + cp["r"] * phi + cp["p"] * psi
            dn_raw = cp["s_dn"] + cp["q"] * omega + cp["p"] * psi

            consequent_outputs.append({
                "mode": np.clip(mode_raw, 0, 2),
                "delta_c": dc_raw,
                "delta_n": dn_raw,
            })
        return consequent_outputs

    def decide(self, psi, omega, phi, rho, n_current, cores_current, predicted_rps,
               ga_checkpoint=None):
        """Layer 5: Weighted summation to produce scaling decision."""
        inputs = {"psi": psi, "omega": omega, "phi": phi, "rho": rho}

        # Layer 1-3: membership -> firing strengths -> normalization
        strengths = self._compute_firing_strengths(inputs)
        w_bar = self._normalize_strengths(strengths)

        # Layer 4: consequent outputs
        consequents = self._compute_consequents(inputs, n_current, cores_current, predicted_rps)

        # Layer 5: weighted summation
        mode_val = sum(w_bar[i] * consequents[i]["mode"] for i in range(len(self.rules)))
        dc_val = sum(w_bar[i] * consequents[i]["delta_c"] for i in range(len(self.rules)))
        dn_val = sum(w_bar[i] * consequents[i]["delta_n"] for i in range(len(self.rules)))

        # Constrain toward NSGA-II Pareto checkpoint if available
        if ga_checkpoint is not None:
            h_star, c_star = ga_checkpoint
            anfis_weight = 1.0 - self._ga_influence
            dn_val = anfis_weight * dn_val + self._ga_influence * (h_star - n_current)
            dc_val = anfis_weight * dc_val + self._ga_influence * (c_star - cores_current)

        # Apply Adaptive Deadzone to prevent micro-oscillations.
        # Deadzone width scales with load: aggressive scale-down at low load,
        # responsive at high load, conservative near capacity.
        if psi < 0.5:
            dz_n = self._dz_low
            dz_c = self._dz_low
        elif psi < 0.8:
            dz_n = self._dz_moderate
            dz_c = self._dz_moderate
        elif psi < 1.1:
            dz_n = self._dz_near_cap
            dz_c = self._dz_near_cap
        else:
            dz_n = self._dz_over_cap
            dz_c = self._dz_over_cap
        
        if abs(dn_val) < dz_n: dn_val = 0.0
        if abs(dc_val) < dz_c: dc_val = 0.0

        # Discretize deltas natively based purely on defuzzification result
        delta_c = int(round(dc_val))
        delta_n = int(round(dn_val))

        # Enforce bounds FIRST before setting mode
        new_cores = np.clip(cores_current + delta_c, self.min_cores, self.max_cores)
        new_replicas = np.clip(n_current + delta_n, self.min_replicas, self.max_replicas)
        delta_c = int(new_cores - cores_current)
        delta_n = int(new_replicas - n_current)

        # Enforce mode dynamically from finalized deltas
        if delta_c != 0 and delta_n != 0:
            mode = "diagonal"
        elif delta_c != 0:
            mode = "vertical"
        elif delta_n != 0:
            mode = "horizontal"
        else:
            mode = "none"

        return {
            "mode": mode,
            "delta_c": delta_c,
            "delta_n": delta_n,
            "firing_strengths": w_bar.tolist(),
            "mode_raw": mode_val,
        }
