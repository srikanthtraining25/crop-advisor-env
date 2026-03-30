# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
Crop Simulator — Models crop growth, soil dynamics, and pest behavior.

Simulates a full growing season with realistic agricultural dynamics:
- Growth stages: seedling → vegetative → flowering → maturity
- Soil moisture depletion and nutrient consumption
- Pest population dynamics with random outbreaks
- Crop health as a function of environmental conditions
"""

import random
from typing import Tuple


# Growth stage thresholds (day ranges)
GROWTH_STAGE_SCHEDULE = [
    (0, 30, "seedling"),
    (30, 75, "vegetative"),
    (75, 130, "flowering"),
    (130, 180, "maturity"),
]

# How much moisture/nutrients the crop consumes per day at each stage
DAILY_CONSUMPTION = {
    "seedling": {"moisture": 0.015, "nutrients": 0.008},
    "vegetative": {"moisture": 0.030, "nutrients": 0.020},
    "flowering": {"moisture": 0.040, "nutrients": 0.025},
    "maturity": {"moisture": 0.020, "nutrients": 0.010},
}

# Optimal ranges for each metric (outside = health penalty)
OPTIMAL_RANGES = {
    "soil_moisture": (0.3, 0.7),
    "soil_nutrients": (0.3, 0.8),
    "pest_level": (0.0, 0.4),
}


class CropSimulator:
    """Simulates crop field dynamics across a growing season."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> dict:
        """Reset to initial field state. Returns the initial state dict."""
        self.day = 0
        self.soil_moisture = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.soil_nutrients = 0.5 + self.rng.uniform(-0.1, 0.1)
        self.pest_level = self.rng.uniform(0.0, 0.05)
        self.crop_health = 1.0
        self.growth_stage = "seedling"
        self.yield_estimate = 0.0
        self.budget = 500.0
        self.initial_budget = 500.0

        return self._get_state_dict()

    def _get_growth_stage(self) -> str:
        """Determine growth stage based on current day."""
        for start, end, stage in GROWTH_STAGE_SCHEDULE:
            if start <= self.day < end:
                return stage
        return "maturity"

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp a value to [low, high]."""
        return max(low, min(high, value))

    def _compute_health_delta(self) -> float:
        """
        Compute health change based on how far soil/pest conditions
        are from optimal ranges.
        """
        delta = 0.0

        # Soil moisture stress
        m_lo, m_hi = OPTIMAL_RANGES["soil_moisture"]
        if self.soil_moisture < m_lo:
            delta -= (m_lo - self.soil_moisture) * 0.15  # water stress
        elif self.soil_moisture > m_hi:
            delta -= (self.soil_moisture - m_hi) * 0.08  # waterlogging

        # Nutrient stress
        n_lo, n_hi = OPTIMAL_RANGES["soil_nutrients"]
        if self.soil_nutrients < n_lo:
            delta -= (n_lo - self.soil_nutrients) * 0.12
        elif self.soil_nutrients > n_hi:
            delta -= (self.soil_nutrients - n_hi) * 0.05  # mild over-fert

        # Pest damage
        p_lo, p_hi = OPTIMAL_RANGES["pest_level"]
        if self.pest_level > p_hi:
            delta -= (self.pest_level - p_hi) * 0.12

        # Natural recovery when conditions are good
        if m_lo <= self.soil_moisture <= m_hi and n_lo <= self.soil_nutrients <= n_hi:
            if self.pest_level <= p_hi:
                delta += 0.03  # natural healing

        return delta

    def _update_yield_estimate(self):
        """Update yield estimate based on crop health and growth stage."""
        stage_multipliers = {
            "seedling": 0.0,
            "vegetative": 0.3,
            "flowering": 0.7,
            "maturity": 1.0,
        }
        multiplier = stage_multipliers.get(self.growth_stage, 0.0)
        # Yield = base * health * stage multiplier (tons/hectare)
        self.yield_estimate = round(5.0 * self.crop_health * multiplier, 2)

    def _simulate_pest_outbreak(self):
        """Random pest outbreak events."""
        # Base daily pest growth
        self.pest_level += self.rng.uniform(0.0, 0.01)

        # Random outbreak (2% chance per day, higher in flowering)
        outbreak_chance = 0.02
        if self.growth_stage == "flowering":
            outbreak_chance = 0.04

        if self.rng.random() < outbreak_chance:
            outbreak_severity = self.rng.uniform(0.05, 0.20)
            self.pest_level += outbreak_severity

        self.pest_level = self._clamp(self.pest_level)

    def apply_action(self, action_type: str, intensity: str) -> Tuple[str, float]:
        """
        Apply an agent's action to the field.

        Args:
            action_type: The action to take.
            intensity: How aggressively to apply it.

        Returns:
            Tuple of (message, cost).
        """
        intensity_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}
        mult = intensity_multiplier.get(intensity, 1.0)

        message = ""
        cost = 0.0

        if action_type == "irrigate":
            water_amount = 0.15 * mult
            self.soil_moisture = self._clamp(self.soil_moisture + water_amount)
            cost = 10.0 * mult
            message = f"Irrigated field (+{water_amount:.2f} moisture). Cost: ${cost:.0f}"

        elif action_type == "fertilize":
            nutrient_amount = 0.15 * mult
            self.soil_nutrients = self._clamp(self.soil_nutrients + nutrient_amount)
            cost = 15.0 * mult
            message = f"Applied fertilizer (+{nutrient_amount:.2f} nutrients). Cost: ${cost:.0f}"

        elif action_type == "apply_pesticide":
            pest_reduction = 0.25 * mult
            self.pest_level = self._clamp(self.pest_level - pest_reduction)
            cost = 20.0 * mult
            message = f"Applied pesticide (-{pest_reduction:.2f} pest level). Cost: ${cost:.0f}"

        elif action_type == "harvest":
            if self.growth_stage == "maturity":
                message = f"Harvested crop! Yield: {self.yield_estimate:.2f} tons/hectare"
            else:
                message = f"Premature harvest at {self.growth_stage} stage! Yield severely reduced."
            cost = 25.0

        elif action_type == "wait":
            message = "Waited and observed the field."
            cost = 0.0

        else:
            message = f"Unknown action: {action_type}. No effect."
            cost = 0.0

        self.budget -= cost
        return message, cost

    def advance_day(self, weather_effects: dict) -> dict:
        """
        Advance the simulation by one day.

        Args:
            weather_effects: Dict with moisture_delta, health_delta, pest_delta
                            from the weather engine.

        Returns:
            Updated state dict.
        """
        self.day += 1
        self.growth_stage = self._get_growth_stage()

        # Apply weather effects
        self.soil_moisture += weather_effects.get("moisture_delta", 0.0)
        self.crop_health += weather_effects.get("health_delta", 0.0)
        self.pest_level += weather_effects.get("pest_delta", 0.0)

        # Daily crop consumption
        consumption = DAILY_CONSUMPTION.get(self.growth_stage, DAILY_CONSUMPTION["seedling"])
        self.soil_moisture -= consumption["moisture"]
        self.soil_nutrients -= consumption["nutrients"]

        # Pest simulation
        self._simulate_pest_outbreak()

        # Health changes from conditions
        health_delta = self._compute_health_delta()
        self.crop_health += health_delta

        # Clamp all values
        self.soil_moisture = self._clamp(self.soil_moisture)
        self.soil_nutrients = self._clamp(self.soil_nutrients)
        self.pest_level = self._clamp(self.pest_level)
        self.crop_health = self._clamp(self.crop_health)

        # Update yield
        self._update_yield_estimate()

        return self._get_state_dict()

    def _get_state_dict(self) -> dict:
        """Return all simulation state as a dict."""
        return {
            "day": self.day,
            "growth_stage": self.growth_stage,
            "soil_moisture": round(self.soil_moisture, 4),
            "soil_nutrients": round(self.soil_nutrients, 4),
            "pest_level": round(self.pest_level, 4),
            "crop_health": round(self.crop_health, 4),
            "yield_estimate": self.yield_estimate,
            "budget_remaining": round(self.budget, 2),
        }

    @property
    def is_crop_dead(self) -> bool:
        """Check if crop health has dropped to zero."""
        return self.crop_health <= 0.0

    @property
    def is_season_over(self) -> bool:
        """Check if the 180-day season is complete."""
        return self.day >= 180

    @property
    def budget_fraction(self) -> float:
        """Return remaining budget as a fraction of initial budget."""
        return self.budget / self.initial_budget if self.initial_budget > 0 else 0.0
