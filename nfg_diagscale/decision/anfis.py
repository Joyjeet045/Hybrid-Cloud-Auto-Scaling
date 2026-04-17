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

        # GA checkpoint influence weight (1 - this = ANFIS weight)
        self._ga_influence = acfg.get("ga_influence_weight", 0.3)

        # Adaptive deadzone widths by load regime
        self._dz_low = acfg.get("deadzone_low", 0.2)
        self._dz_moderate = acfg.get("deadzone_moderate", 0.35)
        self._dz_near_cap = acfg.get("deadzone_near_capacity", 0.5)
        self._dz_over_cap = acfg.get("deadzone_over_capacity", 0.15)

        # Cost normalization factor for online learning
        self._cost_norm = acfg.get("cost_normalization", 0.02)

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
        """
        Hybrid learning: two-phase ANFIS update (Jang 1993).
        Phase 1 (backprop): Updates premise MF params (center, sigma).
        Phase 2 (consequent): Updates first-order Sugeno params (p, q, r)
                               using gradient descent on input-loss correlations.
        """
        if len(self._training_buffer) < 10:
            return

        recent = self._training_buffer[-50:]

        # ── Phase 1: Premise parameter update (backprop on MF centers/sigmas) ──
        for record in recent:
            lat = record["latency"]
            cost = record["cost"]

            slo_loss = max(0.0, lat - self.slo) ** 2
            total_loss = slo_loss + self.alpha_cost * cost

            if total_loss < 1e-6:
                continue

            inputs = record["inputs"]
            for var_name, value in inputs.items():
                if var_name not in self.mf_params:
                    continue
                for term_name, mf_p in self.mf_params[var_name].items():
                    c = mf_p["center"]
                    s = mf_p["sigma"]
                    mu = gaussian_mf(value, c, s, term_name)

                    if mu < 1e-8:
                        continue

                    d_mu_dc = mu * (value - c) / (s ** 2 + 1e-8)
                    d_mu_ds = mu * ((value - c) ** 2) / (s ** 3 + 1e-8)

                    grad_scale = total_loss * 0.001
                    gc = np.clip(grad_scale * d_mu_dc, -1.0, 1.0)
                    gs = np.clip(grad_scale * d_mu_ds, -1.0, 1.0)

                    mf_p["center"] -= self.lr * gc
                    mf_p["sigma"] = np.clip(mf_p["sigma"] - self.lr * gs, 0.05, 5.0)

        # ── Phase 2: Consequent parameter update (first-order Sugeno terms) ──
        # Update p, q, r using simplified gradient descent on the loss:
        # The consequent for each rule contributes f_k = p*psi + q*omega + r*phi + s.
        # df/dp = psi, df/dq = omega, df/dr = phi  (partial derivatives).
        # We update in the direction that reduces SLO-cost loss.
        for record in recent:
            lat = record["latency"]
            # To match Phase 1 logic, we only penalize when lat > slo
            lat_error = max(0.0, lat - self.slo) / max(self.slo, 1.0)
            
            # Normalized cost error using configurable normalization factor
            cost_error = self.alpha_cost * (record["cost"] * self._cost_norm) 

            inputs = record["inputs"]
            psi_val = inputs.get("psi", 1.0)
            omega_val = inputs.get("omega", 0.5)
            phi_val = inputs.get("phi", 0.5)

            # Compute firing strengths to weight each rule's contribution
            strengths = self._compute_firing_strengths(inputs)
            w_bar = self._normalize_strengths(strengths)

            for idx, rule in enumerate(self.rules):
                if w_bar[idx] < 1e-6:
                    continue  # This rule barely fires — skip update

                cp = self.consequent_params[rule.rule_id]
                # Scale update by rule's firing weight (more active rules learn faster)
                lr_scaled = self.lr * 0.1 * w_bar[idx]

                # Gradient: reduce latency error by adjusting consequent params
                # If lat_error > 0, we must strongly increase resources.
                # If lat_error == 0, we slowly decrease resources to save cost.
                if lat_error > 0:
                    grad_signal = cost_error - 5.0 * lat_error
                else:
                    grad_signal = cost_error

                cp["p"] -= lr_scaled * np.clip(grad_signal * psi_val, -1.0, 1.0)
                cp["q"] -= lr_scaled * np.clip(grad_signal * omega_val, -1.0, 1.0)
                cp["r"] -= lr_scaled * np.clip(grad_signal * phi_val, -1.0, 1.0)

                # Clip consequent params to prevent divergence
                cp["p"] = np.clip(cp["p"], -2.0, 2.0)
                cp["q"] = np.clip(cp["q"], -2.0, 2.0)
                cp["r"] = np.clip(cp["r"], -2.0, 2.0)

        self._training_buffer = self._training_buffer[-100:]
