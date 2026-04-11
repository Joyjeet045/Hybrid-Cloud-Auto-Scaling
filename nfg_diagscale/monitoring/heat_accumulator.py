"""Heat-based oscillation suppression."""


class HeatAccumulator:
    def __init__(self, config):
        self.heat_threshold = config["mape_k"]["heat_threshold"]
        self.heat = 0
        self._last_violation = "NONE"

    def update(self, violation_direction):
        """Update heat level based on current violation."""
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
        """Check if change threshold reached."""
        return abs(self.heat) >= self.heat_threshold

    def reset(self):
        """
        Reset heat after a change request is dispatched.
        """
        self.heat = 0

    def get_heat(self):
        return self.heat
