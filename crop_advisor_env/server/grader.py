# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
Grader — Reward computation logic for the CropAdvisor environment.

Computes step-level and episode-level rewards based on:
- Action appropriateness (right action at right time)
- Resource efficiency (avoiding waste)
- Crop outcome (health, yield, harvest success)
- Budget management
"""


class CropGrader:
    """
    Computes rewards for each step and at episode end.

    Reward signals are designed to be:
    - Dense: Frequent feedback for most actions
    - Informative: Higher reward for correct decisions
    - Balanced: Penalties for wasteful/harmful actions
    """

    def compute_step_reward(
        self,
        action_type: str,
        intensity: str,
        soil_moisture_before: float,
        soil_nutrients_before: float,
        pest_level_before: float,
        crop_health_after: float,
        growth_stage: str,
        budget_remaining: float,
    ) -> tuple[float, str]:
        """
        Compute the reward for a single step.

        Returns:
            Tuple of (reward_value, reward_explanation).
        """
        reward = 0.0
        reasons = []

        # --- Base health reward ---
        if crop_health_after > 0.6:
            reward += 1.0
            reasons.append("+1.0 (healthy crop)")
        elif crop_health_after > 0.3:
            reward += 0.3
            reasons.append("+0.3 (moderate health)")
        else:
            reward -= 1.0
            reasons.append("-1.0 (poor crop health)")

        # --- Action-specific rewards ---
        if action_type == "irrigate":
            if soil_moisture_before < 0.3:
                reward += 2.0
                reasons.append("+2.0 (timely irrigation - soil was dry)")
            elif soil_moisture_before > 0.7:
                reward -= 1.5
                reasons.append("-1.5 (wasteful irrigation - soil already wet)")
            else:
                reward += 0.5
                reasons.append("+0.5 (irrigation at moderate moisture)")

        elif action_type == "fertilize":
            if growth_stage in ("vegetative", "flowering"):
                reward += 2.0
                reasons.append(f"+2.0 (fertilizing during {growth_stage} stage)")
            elif growth_stage == "seedling":
                reward += 0.5
                reasons.append("+0.5 (early fertilization)")
            else:
                reward -= 0.5
                reasons.append("-0.5 (late fertilization at maturity)")

            if soil_nutrients_before > 0.8:
                reward -= 3.0
                reasons.append("-3.0 (over-fertilization! nutrients already high)")

        elif action_type == "apply_pesticide":
            if pest_level_before > 0.5:
                reward += 3.0
                reasons.append("+3.0 (critical pest control - high infestation)")
            elif pest_level_before > 0.3:
                reward += 1.5
                reasons.append("+1.5 (preventive pest control)")
            elif pest_level_before < 0.2:
                reward -= 2.0
                reasons.append("-2.0 (unnecessary pesticide - low pest level)")
            else:
                reward += 0.5
                reasons.append("+0.5 (moderate pest control)")

        elif action_type == "wait":
            # Small penalty for waiting when action is needed
            if soil_moisture_before < 0.2:
                reward -= 1.0
                reasons.append("-1.0 (waiting while soil critically dry)")
            elif pest_level_before > 0.6:
                reward -= 1.5
                reasons.append("-1.5 (waiting during severe pest infestation)")
            else:
                reward += 0.2
                reasons.append("+0.2 (patient observation)")

        # --- Budget awareness ---
        if budget_remaining <= 0:
            reward -= 2.0
            reasons.append("-2.0 (budget exhausted!)")

        explanation = " | ".join(reasons)
        return round(reward, 2), explanation

    def compute_harvest_reward(
        self,
        growth_stage: str,
        crop_health: float,
        yield_estimate: float,
    ) -> tuple[float, str]:
        """
        Compute reward for harvest action.

        Returns:
            Tuple of (reward_value, explanation).
        """
        if growth_stage == "maturity":
            # Reward scales with health and yield
            base_reward = 10.0
            health_bonus = crop_health * 40.0  # Up to +40 for perfect health
            reward = base_reward + health_bonus
            return round(reward, 2), f"+{reward:.1f} (successful harvest at maturity! Health: {crop_health:.2f}, Yield: {yield_estimate:.2f})"
        elif growth_stage == "flowering":
            reward = -10.0
            return reward, f"{reward} (premature harvest at flowering - significant yield loss)"
        else:
            reward = -20.0
            return reward, f"{reward} (very premature harvest at {growth_stage} - severe yield loss)"

    def compute_episode_end_reward(
        self,
        harvested: bool,
        crop_died: bool,
        crop_health: float,
        budget_fraction: float,
        yield_estimate: float,
    ) -> tuple[float, str]:
        """
        Compute bonus/penalty at episode end.

        Returns:
            Tuple of (reward_value, explanation).
        """
        reward = 0.0
        reasons = []

        if crop_died:
            reward -= 50.0
            reasons.append("-50.0 (crop died!)")

        if not harvested and not crop_died:
            reward -= 15.0
            reasons.append("-15.0 (season ended without harvesting)")

        if budget_fraction > 0.5:
            reward += 5.0
            reasons.append("+5.0 (budget efficiency bonus: >50% remaining)")
        elif budget_fraction > 0.25:
            reward += 2.0
            reasons.append("+2.0 (decent budget management)")

        explanation = " | ".join(reasons) if reasons else "No end-of-episode bonuses"
        return round(reward, 2), explanation
