"""Adaptive Neuro-Fuzzy Inference System (ANFIS) decision engine."""
import numpy as np
from nfg_diagscale.decision.fuzzy_rules import (
    build_rule_base, gaussian_mf, LINGUISTIC_TERMS, MODE_NAMES,
    SCALING_MODES
)


class ANFISEngine:
    def __init__(self, config):
        self.config = config
        acfg = config["anfis"]
        self.lr = acfg["learning_rate"]
        self.alpha_cost = acfg["alpha_cost_weight"]
        self.slo = config["themis"]["slo_ms"]
        self.pod_max_rps = config["cloud"]["pod_max_rps"]
        self.max_cores = config["cloud"]["max_cores"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.min_cores = config["cloud"]["min_cores"]

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

        self._training_buffer = []

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
        # pods_{t+1} = workload_{t+1} / workload_{pod}
        # For SCALE-UP: use effective_pod_rps (cores * base) to find the minimum feasible replicas
        # given current core count. This correctly reflects that vertical scaling reduces H_needed.
        effective_pod_rps = self.pod_max_rps * cores_current
        n_target_up = max(1, int(np.ceil(predicted_rps / effective_pod_rps)))

        # For SCALE-DOWN: use effective_pod_rps to correctly release pods when cores are high.
        n_target_down = max(1, int(np.ceil(predicted_rps / effective_pod_rps)))

        consequent_outputs = []
        for rule in self.rules:
            cp = self.consequent_params[rule.rule_id]

            psi = inputs.get("psi", 1.0)
            omega = inputs.get("omega", 0.5)
            phi = inputs.get("phi", 0.5)

            # f_i = p_i*psi + q_i*omega + r_i*phi + s_i (first-order Sugeno)
            mode_raw = cp["s_mode"] + cp["p"] * psi + cp["q"] * omega
            dc_raw = cp["s_dc"] + cp["r"] * phi
            dn_raw = cp["s_dn"]

            # Adapt horizontal delta based on pod count gap
            if rule.mode == SCALING_MODES["horizontal"]:
                dn_raw = float(n_target_up - n_current)
            elif rule.mode == SCALING_MODES["diagonal"]:
                # Diagonal: split between V and H
                dn_raw = max(1.0, float(np.ceil(psi / 2.0)))

            # Scale-down rules: use full surplus to aggressively save costs.
            # Maximize cost savings when Psi is low and headroom is ample
            if rule.delta_n < 0:
                # Release all pods that are not required for current predicted RPS
                surplus = n_current - n_target_down
                if surplus > 0:
                    dn_raw = float(-surplus) # Scale down to the bare minimum required
                else:
                    dn_raw = 0.0

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
            # Gentle nudge toward GA target — only 10% influence to
            # prevent over-provisioning bias from Pareto exploration
            dn_val = 0.9 * dn_val + 0.1 * (h_star - n_current)
            dc_val = 0.9 * dc_val + 0.1 * (c_star - cores_current)

        # Apply Adaptive Deadzone to prevent micro-oscillations (Stability vs Cost)
        # Suppress changes to avoid jitter, especially at low loads
        # Increase stability threshold when load is safe (psi < 1.1)
        dz_n = 0.5 if psi < 1.1 else 0.15
        dz_c = 0.5 if psi < 1.1 else 0.15
        
        if abs(dn_val) < dz_n: dn_val = 0.0
        if abs(dc_val) < dz_c: dc_val = 0.0

        # Discretize and clip deltas
        delta_c = int(np.clip(round(dc_val), -2, 4))
        delta_n = int(np.clip(round(dn_val), -3, 10))

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

    def record_outcome(self, inputs, action, latency_observed, cost_observed):
        """
        Record (state, action, outcome) tuple for online ANFIS learning.
        """
        self._training_buffer.append({
            "inputs": inputs,
            "action": action,
            "latency": latency_observed,
            "cost": cost_observed,
        })

    def update_parameters(self):
        """Hybrid learning: backprop updates membership function parameters."""
        if len(self._training_buffer) < 10:
            return

        for record in self._training_buffer[-50:]:
            lat = record["latency"]
            cost = record["cost"]

            # L = (L-SLO)^2 + alpha_cost * Cost
            # We use raw differences as specified; stability is handled by gradient clipping below.
            slo_loss = max(0.0, lat - self.slo) ** 2
            total_loss = slo_loss + self.alpha_cost * cost

            if total_loss < 1e-6:
                continue

            inputs = record["inputs"]
            # Gradient descent on premise parameters
            for var_name, value in inputs.items():
                if var_name not in self.mf_params:
                    continue
                for term_name, mf_p in self.mf_params[var_name].items():
                    c = mf_p["center"]
                    s = mf_p["sigma"]
                    mu = gaussian_mf(value, c, s, term_name)

                    if mu < 1e-8:
                        continue

                    # d(loss)/d(center) via chain rule
                    d_mu_dc = mu * (value - c) / (s ** 2 + 1e-8)
                    # d(loss)/d(sigma)
                    d_mu_ds = mu * ((value - c) ** 2) / (s ** 3 + 1e-8)

                    # Gradient clipping for stability
                    grad_scale = total_loss * 0.001 # Reduced scale
                    gc = np.clip(grad_scale * d_mu_dc, -1.0, 1.0)
                    gs = np.clip(grad_scale * d_mu_ds, -1.0, 1.0)

                    mf_p["center"] -= self.lr * gc
                    mf_p["sigma"] = np.clip(mf_p["sigma"] - self.lr * gs, 0.05, 5.0)

        self._training_buffer = self._training_buffer[-100:]
