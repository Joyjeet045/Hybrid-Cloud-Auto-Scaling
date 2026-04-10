"""
Fuzzy rule base for the ANFIS decision engine.

Rule encoding:
  Each rule maps (Psi, Omega, Phi, rho) -> (mode, delta_c, delta_n)
  Where:
    Psi   = surge ratio (predicted / current RPS)
    Omega = latency headroom (SLO - L_curr) / SLO
    Phi   = vertical headroom 1 - c_curr/c_max
    rho   = SLO violation risk indicator
"""
import numpy as np

SCALING_MODES = {"vertical": 0, "diagonal": 1, "horizontal": 2}
MODE_NAMES = {0: "vertical", 1: "diagonal", 2: "horizontal"}


def gaussian_mf(x, center, sigma, term_name=""):
    """
    Gaussian membership function for ANFIS, modified with
    open-ended shoulders (Z-shaped and S-shaped at domain edges)
    so rule firing doesn't drop to 0 at extreme values.
    """
    # Open-ended left shoulders (Z-shape)
    if term_name in ["low", "tight", "exhausted", "safe"] and x < center:
        return 1.0
    # Open-ended right shoulders (S-shape)
    if term_name in ["critical", "ample", "abundant", "risky"] and x > center:
        return 1.0
        
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2 + 1e-8))


# Linguistic term definitions: (center, sigma) for each variable
# These are initial parameters; ANFIS backprop will tune them.
LINGUISTIC_TERMS = {
    "psi": {
        "low":      (0.7, 0.15),
        "moderate": (1.0, 0.15),
        "high":     (1.5, 0.25),
        "critical": (2.5, 0.5),
    },
    "omega": {
        "tight":    (0.1, 0.1),
        "moderate": (0.4, 0.15),
        "ample":    (0.7, 0.15),
    },
    "phi": {
        "exhausted": (0.1, 0.1),
        "available": (0.5, 0.2),
        "abundant":  (0.8, 0.15),
    },
    "rho": {
        "safe":     (0.0, 0.2),
        "risky":    (1.0, 0.2),
    },
}


class FuzzyRule:
    def __init__(self, rule_id, antecedents, mode, delta_c, delta_n, justification):
        self.rule_id = rule_id
        self.antecedents = antecedents
        self.mode = SCALING_MODES[mode]
        self.delta_c = delta_c
        self.delta_n = delta_n
        self.justification = justification

    def firing_strength(self, inputs):
        """
        Layer 2: rule firing strength = product of membership values.
        """
        strength = 1.0
        for var_name, term_name in self.antecedents.items():
            if var_name not in inputs:
                continue
            terms = LINGUISTIC_TERMS[var_name]
            center, sigma = terms[term_name]
            mu = gaussian_mf(inputs[var_name], center, sigma, term_name)
            strength *= mu
        return strength


def build_rule_base():
    """
    Construct the fuzzy rule base.
    """
    rules = [
        # R0: System stable — no scaling needed.
        # Stability: Penalize disruptive moves.
        # When demand matches capacity and SLO is comfortable, hold steady.
        FuzzyRule(
            "R0",
            {"psi": "moderate", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=0, delta_n=0,
            justification="stability: hold when no stress"
        ),
        # R1: Moderate surge, tight headroom -> vertical scale-up
        FuzzyRule(
            "R1",
            {"psi": "moderate", "omega": "tight"},
            mode="vertical", delta_c=1, delta_n=0,
            justification="vertical-first for moderate surges with SLO pressure"
        ),
        # R2: High surge, vertical available -> diagonal
        FuzzyRule(
            "R2",
            {"psi": "high", "phi": "available"},
            mode="diagonal", delta_c=1, delta_n=1,
            justification="diagonal when both gradients non-zero"
        ),
        # R3: Critical surge — Maximize both axes for instant relief
        FuzzyRule(
            "R3",
            {"psi": "critical", "phi": "available"},
            mode="diagonal", delta_c=4, delta_n=5,
            justification="diagonal-max: heavy load requires instant vertical and planned horizontal boost"
        ),
        # R4: SLO at risk, tight headroom -> extreme diagonal
        FuzzyRule(
            "R4",
            {"rho": "risky", "omega": "tight"},
            mode="diagonal", delta_c=2, delta_n=2,
            justification="emergency-diagonal: rapid relief on both axes"
        ),
        # R5: Low demand, ample headroom -> scale down
        # Hysteresis: only downscale if load is very low to avoid thrashing
        FuzzyRule(
            "R5",
            {"psi": "low", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=-1, delta_n=-1,
            justification="conservative scale-down only when load is definitely low"
        ),
        # R6: Vertical exhausted, high demand -> horizontal add
        FuzzyRule(
            "R6",
            {"psi": "high", "phi": "exhausted"},
            mode="diagonal", delta_c=0, delta_n=4,
            justification="horizontal-burst: maximized replica count when cores are capped"
        ),
    ]
    return rules
