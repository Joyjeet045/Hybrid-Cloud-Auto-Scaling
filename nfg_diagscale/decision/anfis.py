"""
Adaptive Neuro-Fuzzy Inference System (ANFIS) decision engine.

[Jang93] Jang J.-S.R. (1993), "ANFIS: Adaptive-Network-Based Fuzzy
  Inference System", IEEE Trans. Syst. Man Cybern. 23(3):665-685.

  ANFIS Architecture (5 layers):
    Layer 1: Fuzzification - Gaussian membership functions
      mu_i(x) = exp(-(x - c_i)^2 / (2 * sigma_i^2))
    Layer 2: Rule firing strengths - product of membership values
      w_i = prod(mu_ij(x_j))
    Layer 3: Normalized firing strengths
      w_bar_i = w_i / sum(w_k)
    Layer 4: Consequent calculation (Takagi-Sugeno first-order)
      f_i = p_i*x1 + q_i*x2 + r_i
    Layer 5: Output summation
      y = sum(w_bar_i * f_i)

  Hybrid learning (Jang93 sect IV):
    Forward pass: least-squares estimation of consequent parameters
    Backward pass: gradient descent on premise (MF) parameters

Rules sourced from:
  [P1] Themis vertical-first strategy (sect 3, sect 5.2)
  [P3] Diagonal optimality Lemma 1 (sect IV-C)
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
        self.lr = acfg["learning_rate"]
        self.alpha_cost = acfg["alpha_cost_weight"]
        self.slo = config["themis"]["slo_ms"]
        self.pod_max_rps = config["cloud"]["pod_max_rps"]
        self.max_cores = config["cloud"]["max_cores"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.min_cores = config["cloud"]["min_cores"]

        self.rules = build_rule_base()

        # [Jang93] Premise parameters: centers and sigmas for each MF
        self.mf_params = {}
        for var_name, terms in LINGUISTIC_TERMS.items():
            self.mf_params[var_name] = {}
            for term_name, (center, sigma) in terms.items():
                self.mf_params[var_name][term_name] = {
                    "center": center,
                    "sigma": sigma,
                }

        # [Jang93] Consequent parameters for Takagi-Sugeno output
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
        """
        [Jang93] Layer 1: Fuzzification using Gaussian MFs.
        mu_i(x) = exp(-(x - c_i)^2 / (2 * sigma_i^2))
        """
        memberships = {}
        for var_name, value in inputs.items():
            if var_name not in self.mf_params:
                continue
            memberships[var_name] = {}
            for term_name, params in self.mf_params[var_name].items():
                mu = gaussian_mf(value, params["center"], params["sigma"])
                memberships[var_name][term_name] = mu
        return memberships

    def _compute_firing_strengths(self, inputs):
        """
        [Jang93] Layer 2: Rule firing strengths w_i = product of membership values.
        """
        strengths = []
        for rule in self.rules:
            w = 1.0
            for var_name, term_name in rule.antecedents.items():
                if var_name in inputs:
                    params = self.mf_params[var_name][term_name]
                    mu = gaussian_mf(inputs[var_name], params["center"], params["sigma"])
                    w *= mu
            strengths.append(w)
        return np.array(strengths)

    def _normalize_strengths(self, strengths):
        """
        [Jang93] Layer 3: Normalized firing strengths.
        w_bar_i = w_i / sum(w_k)
        """
        total = np.sum(strengths)
        if total < 1e-12:
            return np.ones(len(strengths)) / len(strengths)
        return strengths / total

    def _compute_consequents(self, inputs, n_current, cores_current, predicted_rps):
        """
        [Jang93] Layer 4: Consequent calculation.
        For each rule, compute adaptive output based on Takagi-Sugeno model.

        Scaling magnitudes adapt to current state:
        - [P5 Eq. 7] Horizontal delta from predicted pod count gap
        - Vertical delta from consequent parameters (learned)
        """
        # [P5 Eq. 7] pods_{t+1} = workload_{t+1} / workload_{pod}
        # Pod throughput scales with cores (vertical scaling increases capacity)
        effective_pod_rps = self.pod_max_rps * cores_current
        n_target = max(1, int(np.ceil(predicted_rps / effective_pod_rps)))

        consequent_outputs = []
        for rule in self.rules:
            cp = self.consequent_params[rule.rule_id]

            psi = inputs.get("psi", 1.0)
            omega = inputs.get("omega", 0.5)
            phi = inputs.get("phi", 0.5)

            # [Jang93] f_i = p_i*psi + q_i*omega + r_i*phi + s_i (first-order Sugeno)
            mode_raw = cp["s_mode"] + cp["p"] * psi + cp["q"] * omega
            dc_raw = cp["s_dc"] + cp["r"] * phi
            dn_raw = cp["s_dn"]

            # Adapt horizontal delta based on P5 Eq. 7 pod count gap
            if rule.mode == SCALING_MODES["horizontal"]:
                dn_raw = float(n_target - n_current)
            elif rule.mode == SCALING_MODES["diagonal"]:
                # [P3 Lemma 1] Diagonal: split between V and H
                dn_raw = max(1.0, float(np.ceil(psi / 2.0)))

            # Scale-down rules: delta_n proportional to overcapacity
            if rule.delta_n < 0:
                surplus = n_current - n_target
                if surplus > 0:
                    dn_raw = float(-surplus)
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
        """
        [Jang93] Layer 5: Weighted output summation.
        y = sum(w_bar_i * f_i)

        Full ANFIS forward pass producing scaling decision.
        """
        inputs = {"psi": psi, "omega": omega, "phi": phi, "rho": rho}

        # [Jang93] Layer 1-3: membership -> firing strengths -> normalization
        strengths = self._compute_firing_strengths(inputs)
        w_bar = self._normalize_strengths(strengths)

        # [Jang93] Layer 4: consequent outputs
        consequents = self._compute_consequents(inputs, n_current, cores_current, predicted_rps)

        # [Jang93] Layer 5: weighted summation
        mode_val = sum(w_bar[i] * consequents[i]["mode"] for i in range(len(self.rules)))
        dc_val = sum(w_bar[i] * consequents[i]["delta_c"] for i in range(len(self.rules)))
        dn_val = sum(w_bar[i] * consequents[i]["delta_n"] for i in range(len(self.rules)))

        # Discretize mode
        mode_idx = int(np.clip(round(mode_val), 0, 2))
        mode = MODE_NAMES[mode_idx]

        # Constrain toward NSGA-II Pareto checkpoint if available
        if ga_checkpoint is not None:
            h_star, c_star = ga_checkpoint
            # Gentle nudge toward GA target — only 10% influence to
            # prevent over-provisioning bias from Pareto exploration
            dn_val = 0.9 * dn_val + 0.1 * (h_star - n_current)
            dc_val = 0.9 * dc_val + 0.1 * (c_star - cores_current)

        # Discretize and clip deltas
        delta_c = int(np.clip(round(dc_val), -2, 4))
        delta_n = int(np.clip(round(dn_val), -3, 10))

        # Enforce mode consistency
        if mode == "vertical":
            delta_n = 0
        elif mode == "horizontal":
            delta_c = 0

        # Enforce bounds
        new_cores = np.clip(cores_current + delta_c, self.min_cores, self.max_cores)
        new_replicas = np.clip(n_current + delta_n, self.min_replicas, self.max_replicas)
        delta_c = int(new_cores - cores_current)
        delta_n = int(new_replicas - n_current)

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
        [Jang93 sect IV] Hybrid learning: backward pass updates premise
        parameters (MF centers and sigmas) via gradient descent.

        Loss = sum_t [L_actual - SLO]_+^2 + alpha * Cost_actual
        """
        if len(self._training_buffer) < 10:
            return

        for record in self._training_buffer[-50:]:
            lat = record["latency"]
            cost = record["cost"]

            # Hinge loss: only penalize SLO violations
            slo_loss = max(0.0, lat - self.slo) ** 2
            total_loss = slo_loss + self.alpha_cost * cost

            if total_loss < 1e-6:
                continue

            inputs = record["inputs"]
            # [Jang93] Gradient descent on premise parameters
            for var_name, value in inputs.items():
                if var_name not in self.mf_params:
                    continue
                for term_name, params in self.mf_params[var_name].items():
                    c = params["center"]
                    s = params["sigma"]
                    mu = gaussian_mf(value, c, s)

                    if mu < 1e-8:
                        continue

                    # d(loss)/d(center) via chain rule
                    d_mu_dc = mu * (value - c) / (s ** 2 + 1e-8)
                    # d(loss)/d(sigma)
                    d_mu_ds = mu * ((value - c) ** 2) / (s ** 3 + 1e-8)

                    grad_scale = total_loss * 0.01

                    params["center"] -= self.lr * grad_scale * d_mu_dc
                    params["sigma"] = max(0.01, params["sigma"] - self.lr * grad_scale * d_mu_ds)

        self._training_buffer = self._training_buffer[-100:]
