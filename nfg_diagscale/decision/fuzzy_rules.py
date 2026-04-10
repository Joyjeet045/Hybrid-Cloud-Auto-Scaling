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
        # R3: Critical surge, no vertical room -> horizontal
        FuzzyRule(
            "R3",
            {"psi": "critical", "phi": "exhausted"},
            mode="horizontal", delta_c=0, delta_n=3,
            justification="switch to horizontal at hardware limit"
        ),
        # R4: SLO at risk, tight headroom -> emergency diagonal
        FuzzyRule(
            "R4",
            {"rho": "risky", "omega": "tight"},
            mode="diagonal", delta_c=1, delta_n=1,
            justification="diagonal for dual-axis latency relief"
        ),
        # R5: Low demand, ample headroom -> scale down
        # delta_n is dynamically computed in ANFIS based on overcapacity
        FuzzyRule(
            "R5",
            {"psi": "low", "omega": "ample"},
            mode="vertical", delta_c=-1, delta_n=-1,
            justification="scale-down proportional to overcapacity"
        ),
        # R6: No vertical room, high demand -> horizontal add
        FuzzyRule(
            "R6",
            {"psi": "high", "phi": "exhausted"},
            mode="horizontal", delta_c=0, delta_n=2,
            justification="horizontal when vertical exhausted and demand is high"
        ),
    ]
    return rules
