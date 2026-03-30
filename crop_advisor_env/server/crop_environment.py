# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
CropAdvisorEnvironment — The main OpenEnv Environment implementation.

Orchestrates the weather engine, crop simulator, and grader to provide
a complete RL environment for precision agriculture decision-making.
"""

import uuid
from openenv.core.env_server import Environment

from ..models import CropAction, CropObservation, CropState
from .weather_engine import WeatherEngine
from .crop_simulator import CropSimulator
from .grader import CropGrader


class CropAdvisorEnvironment(Environment):
    """
    RL environment for precision agriculture.

    An agent manages a crop field across a 180-day growing season,
    making daily decisions on irrigation, fertilization, pest control,
    and harvesting to maximize yield while minimizing cost.

    API:
        reset()  → CropObservation (initial state)
        step(action: CropAction) → CropObservation (includes reward)
        state    → CropState (episode metadata)
    """

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.seed = seed
        self.weather = WeatherEngine(seed=seed)
        self.simulator = CropSimulator(seed=seed)
        self.grader = CropGrader()
        self._state = CropState()
        self._done = False
        self._last_reward = 0.0
        self._last_explanation = ""

    def reset(self) -> CropObservation:
        """
        Initialize a new episode.

        Returns:
            CropObservation with the initial field state.
        """
        # Reset components
        sim_state = self.simulator.reset()
        weather_today = self.weather.reset()
        forecast = self.weather.get_forecast(day=0)

        # Reset episode state
        self._state = CropState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_reward=0.0,
            actions_taken=0,
            season_complete=False,
            harvested=False,
            crop_died=False,
        )
        self._done = False
        self._last_reward = 0.0

        return CropObservation(
            day=sim_state["day"],
            growth_stage=sim_state["growth_stage"],
            soil_moisture=sim_state["soil_moisture"],
            soil_nutrients=sim_state["soil_nutrients"],
            pest_level=sim_state["pest_level"],
            crop_health=sim_state["crop_health"],
            weather_today=weather_today,
            weather_forecast=forecast,
            budget_remaining=sim_state["budget_remaining"],
            yield_estimate=sim_state["yield_estimate"],
            message="🌱 New growing season started! Manage your crop wisely.",
            success=True,
        )

    def step(self, action: CropAction) -> CropObservation:
        """
        Execute one day of simulation with the given action.

        Args:
            action: CropAction with action_type and intensity.

        Returns:
            CropObservation with updated field state and reward info.
        """
        try:
            return self._execute_step(action)
        except Exception as e:
            return CropObservation(
                day=self.simulator.day,
                growth_stage=self.simulator.growth_stage,
                soil_moisture=self.simulator.soil_moisture,
                soil_nutrients=self.simulator.soil_nutrients,
                pest_level=self.simulator.pest_level,
                crop_health=self.simulator.crop_health,
                weather_today=self.weather.current_weather,
                weather_forecast=self.weather.get_forecast(self.simulator.day),
                budget_remaining=self.simulator.budget,
                yield_estimate=self.simulator.yield_estimate,
                message=f"Error: {str(e)}",
                success=False,
                error=str(e),
            )

    def _execute_step(self, action: CropAction) -> CropObservation:
        """Internal step execution with full error handling."""

        # Check if episode is already done
        if self._done:
            return CropObservation(
                day=self.simulator.day,
                growth_stage=self.simulator.growth_stage,
                soil_moisture=self.simulator.soil_moisture,
                soil_nutrients=self.simulator.soil_nutrients,
                pest_level=self.simulator.pest_level,
                crop_health=self.simulator.crop_health,
                weather_today=self.weather.current_weather,
                weather_forecast=[],
                budget_remaining=self.simulator.budget,
                yield_estimate=self.simulator.yield_estimate,
                message="⚠️ Episode is over. Call reset() to start a new season.",
                success=False,
                error="Episode already completed",
            )

        # Validate action
        if not action.validate_action():
            return CropObservation(
                day=self.simulator.day,
                growth_stage=self.simulator.growth_stage,
                soil_moisture=self.simulator.soil_moisture,
                soil_nutrients=self.simulator.soil_nutrients,
                pest_level=self.simulator.pest_level,
                crop_health=self.simulator.crop_health,
                weather_today=self.weather.current_weather,
                weather_forecast=self.weather.get_forecast(self.simulator.day),
                budget_remaining=self.simulator.budget,
                yield_estimate=self.simulator.yield_estimate,
                message=f"❌ Invalid action: type='{action.action_type}', intensity='{action.intensity}'",
                success=False,
                error=f"Invalid action_type or intensity",
            )

        # Check budget — if insufficient, force to 'wait' (still advance day)
        action_cost = action.get_cost()
        budget_warning = ""
        if action_cost > self.simulator.budget and action.action_type != "wait":
            budget_warning = f"Budget insufficient for {action.action_type} (need ${action_cost:.0f}, have ${self.simulator.budget:.0f}). Forced to wait. "
            action = CropAction(action_type="wait", intensity="low")

        # Save pre-action state for grading
        moisture_before = self.simulator.soil_moisture
        nutrients_before = self.simulator.soil_nutrients
        pest_before = self.simulator.pest_level

        # --- Apply action ---
        action_msg, cost = self.simulator.apply_action(action.action_type, action.intensity)

        # --- Handle harvest ---
        harvest_reward = 0.0
        harvest_msg = ""
        if action.action_type == "harvest":
            harvest_reward, harvest_msg = self.grader.compute_harvest_reward(
                growth_stage=self.simulator.growth_stage,
                crop_health=self.simulator.crop_health,
                yield_estimate=self.simulator.yield_estimate,
            )
            self._state.harvested = True
            self._done = True
            self._state.season_complete = True

        # --- Advance day (weather + natural dynamics) ---
        if not self._done:
            new_weather = self.weather.next_weather(self.simulator.day)
            weather_effects = self.weather.get_effects(new_weather)
            sim_state = self.simulator.advance_day(weather_effects)
        else:
            new_weather = self.weather.current_weather
            sim_state = self.simulator._get_state_dict()

        # --- Compute step reward ---
        step_reward, reward_explanation = self.grader.compute_step_reward(
            action_type=action.action_type,
            intensity=action.intensity,
            soil_moisture_before=moisture_before,
            soil_nutrients_before=nutrients_before,
            pest_level_before=pest_before,
            crop_health_after=self.simulator.crop_health,
            growth_stage=self.simulator.growth_stage,
            budget_remaining=self.simulator.budget,
        )

        total_step_reward = step_reward + harvest_reward

        # --- Check end conditions ---
        end_reward = 0.0
        end_msg = ""

        if self.simulator.is_crop_dead:
            self._done = True
            self._state.crop_died = True
            self._state.season_complete = True

        if self.simulator.is_season_over and not self._done:
            self._done = True
            self._state.season_complete = True

        if self._done:
            end_reward, end_msg = self.grader.compute_episode_end_reward(
                harvested=self._state.harvested,
                crop_died=self._state.crop_died,
                crop_health=self.simulator.crop_health,
                budget_fraction=self.simulator.budget_fraction,
                yield_estimate=self.simulator.yield_estimate,
            )
            total_step_reward += end_reward

        # --- Update state ---
        self._state.step_count += 1
        self._state.total_reward += total_step_reward
        if action.action_type != "wait":
            self._state.actions_taken += 1
        self._last_reward = total_step_reward

        # --- Build message ---
        messages = []
        if budget_warning:
            messages.append(budget_warning)
        messages.append(action_msg)
        if harvest_msg:
            messages.append(f"🌾 {harvest_msg}")
        messages.append(f"📊 Reward: {total_step_reward:+.2f} ({reward_explanation})")
        if end_msg:
            messages.append(f"🏁 {end_msg}")
        if self._done:
            messages.append(f"📋 Season total reward: {self._state.total_reward:.2f}")

        # Get forecast (empty if done)
        forecast = self.weather.get_forecast(self.simulator.day) if not self._done else []

        return CropObservation(
            day=sim_state["day"],
            growth_stage=sim_state["growth_stage"],
            soil_moisture=sim_state["soil_moisture"],
            soil_nutrients=sim_state["soil_nutrients"],
            pest_level=sim_state["pest_level"],
            crop_health=sim_state["crop_health"],
            weather_today=new_weather,
            weather_forecast=forecast,
            budget_remaining=sim_state["budget_remaining"],
            yield_estimate=sim_state["yield_estimate"],
            message=" | ".join(messages),
            success=True,
        )

    @property
    def state(self) -> CropState:
        """Return current episode state."""
        return self._state

    @property
    def reward(self) -> float:
        """Return the last step's reward."""
        return self._last_reward

    @property
    def done(self) -> bool:
        """Return whether the episode is complete."""
        return self._done
