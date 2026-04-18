"""Fuzzy rule base for the ANFIS decision engine."""
import numpy as np

SCALING_MODES = {"vertical": 0, "diagonal": 1, "horizontal": 2}
MODE_NAMES = {0: "vertical", 1: "diagonal", 2: "horizontal"}


def gaussian_mf(x, center, sigma, term_name=""):
    """Gaussian membership function with open-ended shoulders for edge stability."""
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
        "tight":    (0.25, 0.15), # Increased center: reacts when 25% headroom left (75ms)
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

    def firing_strength(self, inputs):
        """Rule firing strength (product of membership values)."""
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
    
    Rules encode the core NFG-DiagScale policy:
      - Vertical-first: prefer instant core changes to avoid rebalance delay
      - Diagonal coordination: scale both axes when load demands it
      - Proactive cost optimization: scale down aggressively when safe
      - ANFIS online learning fine-tunes these initial consequents
    """
    rules = [
        FuzzyRule(
            "R0",
            {"psi": "moderate", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=0, delta_n=0,
            justification="Stability: hold when load is moderate with ample headroom"
        ),
        # R1: Moderate surge, tight headroom -> vertical scale-up
        FuzzyRule(
            "R1",
            {"psi": "moderate", "omega": "tight"},
            mode="vertical", delta_c=3, delta_n=0,
            justification="vertical-first: instant core boost for moderate surges with SLO pressure"
        ),
        # R2: High surge, vertical available -> diagonal
        FuzzyRule(
            "R2",
            {"psi": "high", "phi": "available"},
            mode="diagonal", delta_c=3, delta_n=1,
            justification="diagonal when both gradients non-zero: vertical-heavy for instant relief"
        ),
        # R3: Critical surge — Maximize both axes for instant relief
        FuzzyRule(
            "R3",
            {"psi": "critical", "phi": "available"},
            mode="diagonal", delta_c=4, delta_n=3,
            justification="diagonal-max: heavy load requires instant vertical and planned horizontal boost"
        ),
        # R4: SLO at risk, tight headroom -> extreme diagonal
        FuzzyRule(
            "R4",
            {"rho": "risky", "omega": "tight"},
            mode="diagonal", delta_c=4, delta_n=2,
            justification="emergency-diagonal: extreme vertical boost to bridge horizontal lag"
        ),
        # R5: Low demand, ample headroom -> aggressive DIAGONAL scale-down
        # Vertical-first scale-down: instant, no rebalance cost
        FuzzyRule(
            "R5",
            {"psi": "low", "omega": "ample", "rho": "safe"},
            mode="vertical", delta_c=-3, delta_n=-2,
            justification="cost-optimize: aggressive vertical+horizontal release when load is confirmed low"
        ),
        # R6: Vertical exhausted, high demand -> horizontal add
        FuzzyRule(
            "R6",
            {"psi": "high", "phi": "exhausted"},
            mode="horizontal", delta_c=0, delta_n=3,
            justification="horizontal-burst: add replicas when cores are capped"
        ),
        # R7: Low demand, abundant vertical headroom -> horizontal cleanup
        FuzzyRule(
            "R7",
            {"psi": "low", "phi": "abundant"},
            mode="horizontal", delta_c=-2, delta_n=-3,
            justification="replica-shed: aggressively remove excess replicas and cores when load is clearly low"
        ),
        # R8: Moderate demand, safe SLO, abundant cores -> cost optimization
        # This rule fills the gap where load is "fine" but we're over-provisioned
        FuzzyRule(
            "R8",
            {"psi": "moderate", "rho": "safe", "phi": "abundant"},
            mode="vertical", delta_c=-2, delta_n=-1,
            justification="cost-trim: reduce over-provisioned resources when current load is easily handled"
        ),
    ]
    return rules

