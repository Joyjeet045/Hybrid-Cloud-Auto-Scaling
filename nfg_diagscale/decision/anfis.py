"""Adaptive Neuro-Fuzzy Inference System (ANFIS) decision engine.

The "Neuro-Fuzzy" core of NF-DiagScale. A zero-order (singleton-consequent)
Takagi-Sugeno fuzzy system (Takagi & Sugeno, 1985) in the five-layer ANFIS
arrangement of Jang (1993): Gaussian membership functions (layer 1), product
T-norm rule firing (layer 2), normalisation (layer 3), singleton (constant)
consequents (layer 4), and the weighted-average defuzzified output (layer 5).

Online self-tuning. Unlike a frozen inference engine, the rule consequents are
adapted online from the *realised* control error via a direct adaptive
fuzzy-control update (Wang, 1993; MIT rule): each interval the controller feeds
back the measured SLO/cost outcome and the engine nudges the fired rules'
singleton consequents (``s_dc``, ``s_dn``) in proportion to their normalised
firing strength. The premise (membership) parameters are held FIXED so the fuzzy
partition stays interpretable, and the consequent update is bounded by a
projection onto ``[-bound, bound]`` plus a deadzone on the error, giving a
stable, deterministic adaptation.
"""
import numpy as np
from nfg_diagscale.decision.fuzzy_rules import (
    build_rule_base, gaussian_mf, LINGUISTIC_TERMS, CALIBRATED_TERMS,
)


class ANFISEngine:
    def __init__(self, config):
        self.config = config
        acfg = config["anfis"]
        self.max_cores = config["cloud"]["max_cores"]
        self.max_replicas = config["cloud"]["max_replicas"]
        self.min_replicas = config["cloud"]["min_replicas"]
        self.min_cores = config["cloud"]["min_cores"]

        self._dz_low = acfg.get("deadzone_low", 0.2)
        self._dz_moderate = acfg.get("deadzone_moderate", 0.35)
        self._dz_near_cap = acfg.get("deadzone_near_capacity", 0.5)
        self._dz_over_cap = acfg.get("deadzone_over_capacity", 0.15)

        adp = config.get("adaptive", {})
        self.adapt_enabled = bool(adp.get("enabled", True))
        self.eta = float(adp.get("eta", 0.10))
        self.adapt_deadzone = float(adp.get("deadzone_eps", 0.05))
        self._bound_dc = float(adp.get("bound_dc", 6.0))
        self._bound_dn = float(adp.get("bound_dn", 5.0))
        self.corrective_weight = float(adp.get("corrective_weight", 0.45))

        self.rules = build_rule_base()

        self._mf_terms = (CALIBRATED_TERMS
                          if acfg.get("calibrated_mf", False)
                          else LINGUISTIC_TERMS)
        self.mf_params = {}
        for var_name, terms in self._mf_terms.items():
            self.mf_params[var_name] = {}
            for term_name, (center, sigma) in terms.items():
                self.mf_params[var_name][term_name] = {
                    "center": center,
                    "sigma": sigma,
                }

        self.consequent_params = {}
        for rule in self.rules:
            self.consequent_params[rule.rule_id] = {
                "s_dc": float(rule.delta_c),
                "s_dn": float(rule.delta_n),
            }
        self._baseline_consequents = {
            rid: dict(cp) for rid, cp in self.consequent_params.items()
        }

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

    def _compute_consequents(self):
        """Layer 4: adaptive Takagi-Sugeno singleton consequents."""
        consequent_outputs = []
        for rule in self.rules:
            cp = self.consequent_params[rule.rule_id]
            consequent_outputs.append({
                "delta_c": cp["s_dc"],
                "delta_n": cp["s_dn"],
            })
        return consequent_outputs

    def decide(self, psi, omega, phi, rho, n_current, cores_current,
               corrective=None):
        """Layer 5: Weighted summation to produce scaling decision."""
        inputs = {"psi": psi, "omega": omega, "phi": phi, "rho": rho}

        strengths = self._compute_firing_strengths(inputs)
        w_bar = self._normalize_strengths(strengths)

        consequents = self._compute_consequents()

        dc_val = sum(w_bar[i] * consequents[i]["delta_c"] for i in range(len(self.rules)))
        dn_val = sum(w_bar[i] * consequents[i]["delta_n"] for i in range(len(self.rules)))

        if corrective is not None and self.corrective_weight > 0.0:
            h_star, c_star = corrective
            w = self.corrective_weight
            dn_val = (1.0 - w) * dn_val + w * (h_star - n_current)
            dc_val = (1.0 - w) * dc_val + w * (c_star - cores_current)

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

        delta_c = int(round(dc_val))
        delta_n = int(round(dn_val))

        new_cores = np.clip(cores_current + delta_c, self.min_cores, self.max_cores)
        new_replicas = np.clip(n_current + delta_n, self.min_replicas, self.max_replicas)
        delta_c = int(new_cores - cores_current)
        delta_n = int(new_replicas - n_current)

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
        }

    def adapt(self, firing_strengths, error):
        """Nudge the fired rules' singleton consequents along a measured error.

        ``error`` is the realised control signal supplied by the controller
        (positive => under-provisioned / scale up; negative => over budget pace /
        scale down). Each rule's two singleton constants ``s_dc`` and ``s_dn`` are
        moved by ``eta * error * w_bar_i`` -- i.e. credit is assigned in
        proportion to how strongly the rule fired on the decision that produced
        this outcome. The update is bounded (projection onto ``[-bound, bound]``)
        and gated by a deadzone on ``|error|`` so tiny, noisy errors do not cause
        chatter. Premise (membership) parameters are never touched.
        """
        if not self.adapt_enabled:
            return
        if not np.isfinite(error) or abs(error) < self.adapt_deadzone:
            return
        w = np.asarray(firing_strengths, dtype=float)
        if w.size != len(self.rules) or not np.isfinite(w).all():
            return
        step = self.eta * float(error)
        for i, rule in enumerate(self.rules):
            cp = self.consequent_params[rule.rule_id]
            cp["s_dc"] = float(np.clip(cp["s_dc"] + step * w[i],
                                       -self._bound_dc, self._bound_dc))
            cp["s_dn"] = float(np.clip(cp["s_dn"] + step * w[i],
                                       -self._bound_dn, self._bound_dn))

    def get_consequents(self):
        """Return the current per-rule singleton consequents (for reporting)."""
        return {
            "rule_id": [r.rule_id for r in self.rules],
            "s_dc": [self.consequent_params[r.rule_id]["s_dc"] for r in self.rules],
            "s_dn": [self.consequent_params[r.rule_id]["s_dn"] for r in self.rules],
        }

    def reset_consequents(self):
        """Restore the seed consequents (called at the start of each episode)."""
        for rid, cp in self._baseline_consequents.items():
            self.consequent_params[rid] = dict(cp)
