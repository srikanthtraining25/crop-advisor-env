"""
CropAdvisor — OpenEnvHack Inference Script

Follows the mandatory stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

from crop_advisor_env.models import CropAction
from crop_advisor_env.server.crop_environment import CropAdvisorEnvironment

# ---------------------------------------------------------------------------
# Environment variables (per hackathon rules)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "kimi-k2.5:cloud")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # if using from_docker_image()

BENCHMARK = "crop_advisor"
MAX_STEPS = 180
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score >= this → success

# ---------------------------------------------------------------------------
# System prompt for the LLM
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert agricultural AI managing a farm for a 180-day season.

    You receive daily farm observations containing moisture, pests, nutrients,
    health, and a 3-day weather forecast.

    Valid actions (return EXACTLY this JSON structure, no markdown):
    {"action_type": "irrigate", "intensity": "medium"}
    {"action_type": "fertilize", "intensity": "medium"}
    {"action_type": "apply_pesticide", "intensity": "high"}
    {"action_type": "harvest", "intensity": "medium"}
    {"action_type": "wait", "intensity": "low"}

    Important Rules:
    - Irrigate if soil_moisture < 0.35, but check for rain!
    - Apply pesticide if pest_level > 0.3
    - Fertilize if soil_nutrients < 0.4
    - Harvest ALWAYS when growth_stage is 'maturity' (day >= 135) to get massive reward.
""")


# ---------------------------------------------------------------------------
# Mandatory stdout logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------
def parse_action(response_text: str) -> dict:
    """Parse the LLM response into an action dict; fallback to wait."""
    try:
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
        return json.loads(text)
    except Exception:
        return {"action_type": "wait", "intensity": "low"}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def get_model_action(client: OpenAI, obs_json: str) -> str:
    """Call the LLM and return raw response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_json},
            ],
            temperature=0.0,
            max_tokens=50,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else '{"action_type": "wait", "intensity": "low"}'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '{"action_type": "wait", "intensity": "low"}'


# ---------------------------------------------------------------------------
# Run a single task (episode)
# ---------------------------------------------------------------------------
def run_task(client: OpenAI, task_id: str, seed: int) -> float:
    """Run one episode and return the normalized score in [0, 1]."""
    env = CropAdvisorEnvironment(seed=seed)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            # Build observation JSON for the LLM
            obs_json = json.dumps(obs.model_dump())

            # Get LLM decision
            response_text = get_model_action(client, obs_json)
            action_dict = parse_action(response_text)
            action = CropAction(**action_dict)

            # Take step
            prev_total = env.state.total_reward
            obs = env.step(action)
            step_reward = env.state.total_reward - prev_total
            done = env.done
            error = obs.error if obs.error else None

            rewards.append(step_reward)
            steps_taken = step

            # Format action string for logging
            action_str = f"{action.action_type}({action.intensity})"
            log_step(step=step, action=action_str, reward=step_reward, done=done, error=error)

            if done:
                break

        # Normalize score to (0, 1) — validator rejects exactly 0.0 and 1.0
        score = env.state.total_reward / 100.0
        score = min(max(score, 0.01), 0.99)  # clamp to strict (0, 1)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        # env has no close() method, but [END] must ALWAYS be emitted
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "dummy_key")

    # Tasks defined in openenv.yaml
    tasks = [
        ("task_survive_drought", 100),
        ("task_ideal_conditions", 200),
        ("task_severe_pests", 300),
    ]

    for task_id, seed in tasks:
        run_task(client, task_id, seed)


if __name__ == "__main__":
    main()
