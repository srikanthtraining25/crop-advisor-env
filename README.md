# 🌾 CropAdvisor RL Environment

**An OpenEnv environment for precision agriculture — train AI agents to manage crop fields.**

Built with [Meta's OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework for the Meta PyTorch OpenEnv Hackathon.

---

## Overview

CropAdvisor simulates a **180-day crop growing season** where an RL agent makes daily decisions:

| Decision | Options |
|----------|---------|
| 🚿 Irrigate | Low / Medium / High intensity |
| 🧪 Fertilize | Low / Medium / High intensity |
| 🐛 Apply Pesticide | Low / Medium / High intensity |
| 🌾 Harvest | Ends the episode |
| ⏳ Wait | No action, observe |

The agent must balance **crop health**, **resource costs**, and **timing** to maximize yield.

### Key Features

- **Stochastic weather** — Markov-chain model with seasonal variation (drought, storms, rain)
- **Growth stages** — Seedling → Vegetative → Flowering → Maturity
- **Pest outbreaks** — Random infestations requiring timely intervention
- **Budget management** — Each action costs money; budget is limited
- **Dense rewards** — Immediate feedback on every action + episode-end bonuses

---

## Quick Start

### Install

```bash
pip install openenv-core
pip install git+https://huggingface.co/spaces/openenv/crop-advisor-env
```

### Usage (Sync)

```python
from crop_advisor_env import CropAction, CropAdvisorEnv

with CropAdvisorEnv(base_url="https://openenv-crop-advisor-env.hf.space").sync() as env:
    # Start a new season
    obs = env.reset()
    print(f"Day {obs.day}: {obs.message}")

    # Irrigate the field
    obs = env.step(CropAction(action_type="irrigate", intensity="medium"))
    print(f"Day {obs.day}: moisture={obs.soil_moisture:.2f}, health={obs.crop_health:.2f}")
    print(f"Feedback: {obs.message}")

    # Fertilize during vegetative stage
    obs = env.step(CropAction(action_type="fertilize", intensity="high"))
    print(f"Budget remaining: ${obs.budget_remaining:.0f}")
```

### Usage (Async)

```python
import asyncio
from crop_advisor_env import CropAction, CropAdvisorEnv

async def main():
    async with CropAdvisorEnv(base_url="https://openenv-crop-advisor-env.hf.space") as env:
        obs = await env.reset()
        obs = await env.step(CropAction(action_type="irrigate", intensity="medium"))
        print(obs.message)

asyncio.run(main())
```

---

## Action / Observation Specification

### CropAction

| Field | Type | Values |
|-------|------|--------|
| `action_type` | `str` | `"irrigate"`, `"fertilize"`, `"apply_pesticide"`, `"harvest"`, `"wait"` |
| `intensity` | `str` | `"low"`, `"medium"`, `"high"` |

### CropObservation

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `day` | `int` | 0–180 | Current day in the season |
| `growth_stage` | `str` | — | `"seedling"`, `"vegetative"`, `"flowering"`, `"maturity"` |
| `soil_moisture` | `float` | 0.0–1.0 | Soil water level |
| `soil_nutrients` | `float` | 0.0–1.0 | Soil nutrient level |
| `pest_level` | `float` | 0.0–1.0 | Pest infestation severity |
| `crop_health` | `float` | 0.0–1.0 | Overall crop health |
| `weather_today` | `str` | — | `"sunny"`, `"cloudy"`, `"rainy"`, `"drought"`, `"storm"` |
| `weather_forecast` | `list[str]` | — | Next 3 days forecast |
| `budget_remaining` | `float` | — | Money left ($) |
| `yield_estimate` | `float` | — | Estimated yield (tons/ha) |
| `message` | `str` | — | Feedback on last action |

### CropState

| Field | Type | Description |
|-------|------|-------------|
| `episode_id` | `str` | Unique episode identifier |
| `step_count` | `int` | Steps taken this episode |
| `total_reward` | `float` | Accumulated reward |
| `harvested` | `bool` | Whether crop was harvested |
| `crop_died` | `bool` | Whether crop died |

---

## Reward Structure

| Signal | Reward | Condition |
|--------|--------|-----------|
| Healthy crop | +1.0/step | crop_health > 0.6 |
| Timely irrigation | +2.0 | Irrigate when moisture < 0.3 |
| Wasteful irrigation | -1.5 | Irrigate when moisture > 0.7 |
| Good fertilization | +2.0 | Fertilize at vegetative/flowering |
| Over-fertilization | -3.0 | Nutrients already > 0.8 |
| Critical pest control | +3.0 | Pesticide when pests > 0.5 |
| Unnecessary pesticide | -2.0 | Pesticide when pests < 0.2 |
| Successful harvest | +10 to +50 | Harvest at maturity |
| Premature harvest | -20 | Harvest before maturity |
| Crop death | -50 | Health drops to 0 |
| Budget bonus | +5.0 | End with >50% budget |

---

## Build & Run Locally

```bash
# Install in development mode
pip install -e ".[dev]"

# Run server locally
uvicorn crop_advisor_env.server.app:app --host 0.0.0.0 --port 8000

# Run strategy tests
python tests/test_all_strategies.py
```

### Run Hackathon Inference Script
The `inference.py` script is fully compliant with Hackathon submission guidelines:
```bash
# Configures the LLM endpoint internally
python inference.py
```

### Docker

```bash
docker build -t crop-advisor-env .
docker run -p 8000:8000 crop-advisor-env
```

---

## License

BSD 3-Clause License
