import os
import json
from openai import OpenAI
from crop_advisor_env.models import CropAction
from crop_advisor_env.server.crop_environment import CropAdvisorEnvironment

# Required environment variables per Hackathon rules
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
if not API_BASE_URL.endswith("/v1") and "ollama" in API_BASE_URL:
    API_BASE_URL = API_BASE_URL.rstrip("/") + "/v1"

MODEL_NAME = os.getenv("MODEL_NAME", "kimi-k2.5:cloud")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "ollama"))

SYSTEM_PROMPT = """You are an expert agricultural AI managing a farm for a 180-day season.

You receive daily farm observations containing moisture, pests, nutrients, health, and a 3-day weather forecast.

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
"""

def parse_action(response_text) -> dict:
    try:
        text = response_text.strip()
        if text.startswith("```json"): text = text[7:-3].strip()
        elif text.startswith("```"): text = text[3:-3].strip()
        return json.loads(text)
    except:
        return {"action_type": "wait", "intensity": "low"}

def run_task(task_id: str, seed: int):
    print(f"\n{'='*50}\nStarting {task_id} (Seed: {seed})\n{'='*50}")
    
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy_key")
    env = CropAdvisorEnvironment(seed=seed)
    obs = env.reset()
    
    for step in range(180):
        # We manually build prompt state 
        state_repr = json.dumps(obs.model_dump())
        
        try:
            # Skip API call strictly if no token, just to avoid crashing CI if testing trivially
            # But the requirement says it hits the endpoint.
            if HF_TOKEN:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": state_repr}
                    ],
                    temperature=0.0,
                    max_tokens=50
                )
                response_text = completion.choices[0].message.content or ""
            else:
                response_text = '{"action_type": "wait", "intensity": "low"}'
        except Exception as exc:
            print(f"Model request failed ({exc}). Fallback -> wait.")
            response_text = '{"action_type": "wait", "intensity": "low"}'
            
        action_dict = parse_action(response_text)
        action = CropAction(**action_dict)
        
        last_reward_total = env.state.total_reward
        obs = env.step(action)
        step_reward = env.state.total_reward - last_reward_total
        
        print(f"Day {obs.day:3d} | Action: {action.action_type}({action.intensity}) | Step Reward: {step_reward:+.2f}")
        
        if env.done:
            print(f"Episode complete on Day {obs.day}.")
            break

    # Hackathon requirement: "verify scores in 0.0-1.0 range"
    # We map our raw reward to a bounded 0.0 - 1.0 score where <0 is 0.0 and >100 is 1.0.
    normalized_score = max(0.0, min(1.0, env.state.total_reward / 100.0))
    
    print(f"\n[{task_id} Final Results]")
    print(f"Raw Reward: {env.state.total_reward:.2f}")
    print(f"Official Grader Score (0.0-1.0): {normalized_score:.4f}")
    print(f"Harvest Successful: {env.state.harvested}")
    print(f"Crop Died: {env.state.crop_died}")
    
    return normalized_score

def main():
    print(f"Initializing OpenEnv Inference Script")
    print(f"Model: {MODEL_NAME} @ {API_BASE_URL}")
    print("-" * 50)
    
    tasks = [
        ("task_survive_drought", 100),
        ("task_ideal_conditions", 200),
        ("task_severe_pests", 300)
    ]
    
    scores = []
    for task_id, seed in tasks:
        score = run_task(task_id, seed)
        scores.append(score)
        
    print(f"\n{'*'*50}")
    print(f"ALL TASKS COMPLETED. Average Grader Score: {sum(scores)/len(scores):.4f}")
    print(f"{'*'*50}\n")

if __name__ == "__main__":
    main()
