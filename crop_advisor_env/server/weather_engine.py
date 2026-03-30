# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
Weather Engine — Stochastic Markov-chain weather simulation.

Generates realistic weather sequences with seasonal variation.
Weather affects soil moisture, crop health, and pest dynamics.
"""

import random
from typing import List


# Markov transition probabilities: weather_today → {weather_tomorrow: probability}
# These shift with the season to simulate realistic patterns.

WEATHER_TRANSITIONS = {
    "sunny": {"sunny": 0.40, "cloudy": 0.30, "rainy": 0.15, "drought": 0.10, "storm": 0.05},
    "cloudy": {"sunny": 0.25, "cloudy": 0.30, "rainy": 0.30, "drought": 0.05, "storm": 0.10},
    "rainy": {"sunny": 0.15, "cloudy": 0.30, "rainy": 0.30, "drought": 0.02, "storm": 0.23},
    "drought": {"sunny": 0.35, "cloudy": 0.20, "rainy": 0.05, "drought": 0.35, "storm": 0.05},
    "storm": {"sunny": 0.20, "cloudy": 0.35, "rainy": 0.30, "drought": 0.02, "storm": 0.13},
}

# Seasonal modifiers: boost or reduce certain weather probabilities
# Early season (day 0-60): more rain, mid (60-120): mixed, late (120-180): drier
SEASONAL_MODIFIERS = {
    "early": {"rainy": 1.3, "drought": 0.5, "storm": 0.8},
    "mid": {"rainy": 1.0, "drought": 1.0, "storm": 1.2},
    "late": {"rainy": 0.6, "drought": 1.8, "storm": 0.7},
}

# Impact of weather on environment
WEATHER_EFFECTS = {
    "sunny": {"moisture_delta": -0.05, "health_delta": 0.01, "pest_delta": 0.01},
    "cloudy": {"moisture_delta": -0.02, "health_delta": 0.005, "pest_delta": 0.005},
    "rainy": {"moisture_delta": 0.15, "health_delta": 0.01, "pest_delta": -0.02},
    "drought": {"moisture_delta": -0.12, "health_delta": -0.05, "pest_delta": 0.03},
    "storm": {"moisture_delta": 0.20, "health_delta": -0.08, "pest_delta": -0.05},
}


class WeatherEngine:
    """Stochastic weather simulation using Markov chains with seasonal variation."""

    def __init__(self, seed: int | None = None):
        """
        Initialize the weather engine.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = random.Random(seed)
        self.current_weather = "sunny"

    def reset(self) -> str:
        """Reset weather to a random starting state."""
        self.current_weather = self.rng.choice(["sunny", "cloudy", "rainy"])
        return self.current_weather

    def _get_season(self, day: int) -> str:
        """Determine the season phase based on the day."""
        if day < 60:
            return "early"
        elif day < 120:
            return "mid"
        else:
            return "late"

    def _apply_seasonal_modifier(self, transitions: dict, day: int) -> dict:
        """Apply seasonal modifiers to transition probabilities."""
        season = self._get_season(day)
        modifiers = SEASONAL_MODIFIERS[season]

        modified = {}
        for weather, prob in transitions.items():
            modifier = modifiers.get(weather, 1.0)
            modified[weather] = prob * modifier

        # Normalize so probabilities sum to 1
        total = sum(modified.values())
        return {w: p / total for w, p in modified.items()}

    def next_weather(self, day: int) -> str:
        """
        Generate the next day's weather.

        Args:
            day: Current day number (0-180).

        Returns:
            The new weather condition string.
        """
        base_transitions = WEATHER_TRANSITIONS[self.current_weather]
        transitions = self._apply_seasonal_modifier(base_transitions, day)

        weather_options = list(transitions.keys())
        probabilities = list(transitions.values())

        # Weighted random choice
        self.current_weather = self.rng.choices(weather_options, weights=probabilities, k=1)[0]
        return self.current_weather

    def get_forecast(self, day: int, days_ahead: int = 3) -> List[str]:
        """
        Generate a weather forecast for the next N days.

        The forecast is probabilistic (not guaranteed accurate) — adds
        some uncertainty to simulate real forecasting.

        Args:
            day: Current day.
            days_ahead: Number of days to forecast.

        Returns:
            List of predicted weather conditions.
        """
        forecast = []
        # Save current state
        saved_weather = self.current_weather
        saved_rng_state = self.rng.getstate()

        for i in range(days_ahead):
            predicted = self.next_weather(day + i + 1)

            # Add forecast uncertainty: 20% chance of being wrong
            if self.rng.random() < 0.20:
                alternatives = [w for w in WEATHER_TRANSITIONS.keys() if w != predicted]
                predicted = self.rng.choice(alternatives)

            forecast.append(predicted)

        # Restore state (forecast shouldn't affect actual weather)
        self.current_weather = saved_weather
        self.rng.setstate(saved_rng_state)

        return forecast

    def get_effects(self, weather: str) -> dict:
        """
        Get the environmental effects for a given weather condition.

        Returns:
            Dict with moisture_delta, health_delta, pest_delta.
        """
        return WEATHER_EFFECTS.get(weather, WEATHER_EFFECTS["sunny"]).copy()
