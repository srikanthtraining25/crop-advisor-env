# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
CropAdvisor Environment Models

Defines the Action, Observation, and State models for the
CropAdvisor RL environment, inheriting from OpenEnv base classes.

Note: OpenEnv base classes (Action, Observation, State) are Pydantic BaseModel
subclasses, so we use Pydantic field definitions here.
"""

from typing import List, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


# --- Valid constants ---

VALID_ACTIONS = ["irrigate", "fertilize", "apply_pesticide", "harvest", "wait"]
VALID_INTENSITIES = ["low", "medium", "high"]
VALID_GROWTH_STAGES = ["seedling", "vegetative", "flowering", "maturity"]
VALID_WEATHER = ["sunny", "cloudy", "rainy", "drought", "storm"]

# --- Cost tables ---

ACTION_COSTS = {
    "irrigate": {"low": 5.0, "medium": 10.0, "high": 20.0},
    "fertilize": {"low": 8.0, "medium": 15.0, "high": 30.0},
    "apply_pesticide": {"low": 10.0, "medium": 20.0, "high": 40.0},
    "harvest": {"low": 25.0, "medium": 25.0, "high": 25.0},
    "wait": {"low": 0.0, "medium": 0.0, "high": 0.0},
}


class CropAction(Action):
    """
    An action the agent takes each day on the crop field.

    Attributes:
        action_type: One of "irrigate", "fertilize", "apply_pesticide", "harvest", "wait"
        intensity: Resource usage level - "low", "medium", "high"
    """
    action_type: str = Field(default="wait", description="Action to take on the crop field")
    intensity: str = Field(default="medium", description="Resource usage level")

    def validate_action(self) -> bool:
        """Check that action_type and intensity are valid."""
        return (
            self.action_type in VALID_ACTIONS
            and self.intensity in VALID_INTENSITIES
        )

    def get_cost(self) -> float:
        """Return the monetary cost for this action + intensity."""
        return ACTION_COSTS.get(self.action_type, {}).get(self.intensity, 0.0)


class CropObservation(Observation):
    """
    What the agent sees after each step.

    Attributes:
        day: Current day in the season (0-180)
        growth_stage: Current crop growth stage
        soil_moisture: Soil water level (0.0-1.0)
        soil_nutrients: Soil nutrient level (0.0-1.0)
        pest_level: Severity of pest infestation (0.0-1.0)
        crop_health: Overall crop health (0.0-1.0)
        weather_today: Current day's weather condition
        weather_forecast: Forecast for the next 3 days
        budget_remaining: Money left for the season ($)
        yield_estimate: Estimated harvest yield (tons/hectare)
        message: Human-readable feedback on the last action
        success: Whether the last action succeeded
        error: Error description if action failed
    """
    day: int = Field(default=0, description="Current day in season (0-180)")
    growth_stage: str = Field(default="seedling", description="Current growth stage")
    soil_moisture: float = Field(default=0.5, description="Soil water level 0.0-1.0")
    soil_nutrients: float = Field(default=0.5, description="Soil nutrient level 0.0-1.0")
    pest_level: float = Field(default=0.0, description="Pest infestation 0.0-1.0")
    crop_health: float = Field(default=1.0, description="Overall crop health 0.0-1.0")
    weather_today: str = Field(default="sunny", description="Today's weather")
    weather_forecast: List[str] = Field(default_factory=lambda: ["sunny", "sunny", "sunny"], description="Next 3 days forecast")
    budget_remaining: float = Field(default=500.0, description="Money remaining ($)")
    yield_estimate: float = Field(default=0.0, description="Estimated yield (tons/ha)")
    message: str = Field(default="", description="Feedback on last action")
    success: bool = Field(default=True, description="Whether last action succeeded")
    error: str = Field(default="", description="Error description if failed")


class CropState(State):
    """
    Internal episode state tracking.

    Extends OpenEnv's base State (which provides episode_id, step_count).

    Attributes:
        total_reward: Accumulated reward for this episode
        actions_taken: Count of non-wait actions taken
        season_complete: Whether the season ended
        harvested: Whether the crop was successfully harvested
        crop_died: Whether the crop died during the season
    """
    total_reward: float = Field(default=0.0, description="Accumulated reward")
    actions_taken: int = Field(default=0, description="Non-wait actions taken")
    season_complete: bool = Field(default=False, description="Whether season ended")
    harvested: bool = Field(default=False, description="Whether crop was harvested")
    crop_died: bool = Field(default=False, description="Whether crop died")
