"""
Heat-based oscillation suppression.

[P4] Solino, Batista & Cavalcante (2025), "An Autonomic Computing Approach
  for Scaling Cloud-Based Smart City Platforms", ACM UCC'25

  Algorithm 3 (P4 sect 2.2): ScalingHeat procedure.
  "The concept of heat determines the accumulation of resources to add
   or remove. If the upper/lower limit is violated, the heat
   increases/decreases. When no threshold violation occurs, the heat
   gradually returns to zero."

  [P4 Table 1] heatThreshold parameter controls when a change request
  is dispatched. Default value = 3.
"""


class HeatAccumulator:
    def __init__(self, config):
        # [P4 Table 1] heatThreshold parameter
        self.heat_threshold = config["mape_k"]["heat_threshold"]
        self.heat = 0
        self._last_violation = "NONE"

    def update(self, violation_direction):
        """
        [P4 Algorithm 3] ScalingHeat procedure.

        violation_direction: "UP", "DOWN", or "NONE"

        Lines 5-16 of P4 Algorithm 3:
        - If violation = UP and heat <= 0: heat = 1
        - If violation = UP and heat > 0:  heat = heat + 1
        - If violation = DOWN and heat >= 0: heat = -1
        - If violation = DOWN and heat < 0:  heat = heat - 1
        - If violation = NONE and heat > 0: heat = heat - 1
        - If violation = NONE and heat < 0: heat = heat + 1
        - If violation = NONE and heat == 0: heat = 0
        """
        if violation_direction == "UP":
            # [P4 Alg. 3 lines 5-10]
            if self.heat <= 0:
                self.heat = 1
            else:
                self.heat = self.heat + 1

        elif violation_direction == "DOWN":
            # [P4 Alg. 3 lines 11-16]
            if self.heat >= 0:
                self.heat = -1
            else:
                self.heat = self.heat - 1

        else:
            # [P4 Alg. 3 lines 17-24] NONE: cool down toward zero
            if self.heat > 0:
                self.heat = self.heat - 1
            elif self.heat < 0:
                self.heat = self.heat + 1

        self._last_violation = violation_direction

    def should_trigger(self):
        """
        [P4 Algorithm 4] sendChangeRequest procedure.
        "A change request is sent to the Plan phase when the value
         of the heat attribute reaches the threshold value defined
         by the heatThreshold parameter."
        """
        return abs(self.heat) >= self.heat_threshold

    def reset(self):
        """
        [P4 Alg. 4 line 5] "CR.heat <- 0"
        Reset heat after a change request is dispatched.
        """
        self.heat = 0

    def get_heat(self):
        return self.heat
