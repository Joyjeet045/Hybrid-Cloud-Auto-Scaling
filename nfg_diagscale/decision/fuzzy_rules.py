"""Fuzzy rule base for the ANFIS decision engine."""
import numpy as np

SCALING_MODES = {"vertical": 0, "diagonal": 1, "horizontal": 2}
MODE_NAMES = {0: "vertical", 1: "diagonal", 2: "horizontal"}


def gaussian_mf(x, center, sigma, term_name=""):
    """Gaussian membership function with open-ended shoulders for edge stability."""
    if term_name in ["low", "tight", "exhausted", "safe"] and x < center:
        return 1.0
    if term_name in ["critical", "ample", "abundant", "risky"] and x > center:
        return 1.0
        
    return np.exp(-((x - center) ** 2) / (2 * sigma ** 2 + 1e-8))


LINGUISTIC_TERMS = {
    "psi": {
        "low":      (0.7, 0.15),
        "moderate": (1.0, 0.15),
        "high":     (1.5, 0.25),
        "critical": (2.5, 0.5),
    },
    "omega": {
        "tight":    (0.25, 0.15),
        "moderate": (0.5, 0.15),
        "ample":    (0.8, 0.15),
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


def build_rule_base():
    """
    Construct the fuzzy rule base.
    
    Rules encode the core NF-DiagScale policy:
      - Vertical-first: prefer instant core changes to avoid rebalance delay
      - Diagonal coordination: scale both axes when load demands it
      - Proactive cost optimization: scale down aggressively when safe
      - Online self-tuning adapts these singleton consequents from the realised
        SLO/cost feedback (the seed magnitudes below avoid a cold start)
    """
    rules = [
        FuzzyRule(
            "R0",
            {"psi": "moderate", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=0, delta_n=0,
            justification="Stability: hold when load is moderate with ample headroom"
        ),
        FuzzyRule(
            "R1",
            {"psi": "moderate", "omega": "tight"},
            mode="vertical", delta_c=3, delta_n=0,
            justification="vertical-first: instant core boost for moderate surges with SLO pressure"
        ),
        FuzzyRule(
            "R2",
            {"psi": "high", "phi": "available"},
            mode="diagonal", delta_c=3, delta_n=1,
            justification="diagonal when both gradients non-zero: vertical-heavy for instant relief"
        ),
        FuzzyRule(
            "R3",
            {"psi": "critical", "phi": "available"},
            mode="diagonal", delta_c=4, delta_n=3,
            justification="diagonal-max: heavy load requires instant vertical and planned horizontal boost"
        ),
        FuzzyRule(
            "R4",
            {"rho": "risky", "omega": "tight"},
            mode="diagonal", delta_c=4, delta_n=2,
            justification="emergency-diagonal: extreme vertical boost to bridge horizontal lag"
        ),
        FuzzyRule(
            "R5",
            {"psi": "low", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=-3, delta_n=-2,
            justification="cost-optimize: aggressive vertical+horizontal release when load is confirmed low"
        ),
        FuzzyRule(
            "R6",
            {"psi": "high", "phi": "exhausted"},
            mode="horizontal", delta_c=0, delta_n=3,
            justification="horizontal-burst: add replicas when cores are capped"
        ),
        FuzzyRule(
            "R7",
            {"psi": "low", "phi": "abundant"},
            mode="horizontal", delta_c=-2, delta_n=-3,
            justification="replica-shed: aggressively remove excess replicas and cores when load is clearly low"
        ),
        FuzzyRule(
            "R8",
            {"psi": "moderate", "rho": "safe", "phi": "abundant"},
            mode="vertical", delta_c=-2, delta_n=-1,
            justification="cost-trim: reduce over-provisioned resources when current load is easily handled"
        ),
    ]
    return rules

