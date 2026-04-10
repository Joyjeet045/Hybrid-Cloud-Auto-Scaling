"""
Heat-based oscillation suppression.

The concept of heat determines the accumulation of resources to add
or remove. If the limit is violated, the heat increases/decreases. 
When no violation occurs, the heat gradually returns to zero.
"""


class HeatAccumulator:
    def __init__(self, config):
        self.heat_threshold = config["mape_k"]["heat_threshold"]
        self.heat = 0
        self._last_violation = "NONE"

    def update(self, violation_direction):
        """
        Update the heat level based on current violation direction.

        violation_direction: "UP", "DOWN", or "NONE"
        """
        if violation_direction == "UP":
            if self.heat <= 0:
                self.heat = 1
            else:
                self.heat = self.heat + 1

        elif violation_direction == "DOWN":
            if self.heat >= 0:
                self.heat = -1
            else:
                self.heat = self.heat - 1

        else:
            # NONE: cool down toward zero
            if self.heat > 0:
                self.heat = self.heat - 1
            elif self.heat < 0:
                self.heat = self.heat + 1

        self._last_violation = violation_direction

    def should_trigger(self):
        """
        Determine if the heat threshold has been reached to trigger a change.
        """
        return abs(self.heat) >= self.heat_threshold

    def reset(self):
        """
        Reset heat after a change request is dispatched.
        """
        self.heat = 0

    def get_heat(self):
        return self.heat
