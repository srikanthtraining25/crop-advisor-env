# CropAdvisor RL Environment — Test Cases & Agent Strategies Guide

## Table of Contents

1. [Environment Understanding](#environment-understanding)
2. [Agent Strategy Approaches](#agent-strategy-approaches)
3. [Test Scenarios](#test-scenarios)
4. [Running Tests](#running-tests)
5. [Expected Outcomes](#expected-outcomes)

---

## Environment Understanding

### Episode Lifecycle

```
Day 0 (reset) ──► Agent action ──► Weather event ──► Soil/Crop update ──► Reward ──► Day+1
                      │                                                        │
                      └────────────────── repeat until ────────────────────────┘
                                                                               │
                                                           done when: day≥180 | crop_dead | harvested
```

### What Makes or Breaks a Season

| Factor | Impact | How to Manage |
|--------|--------|--------------|
| **Soil moisture** | Below 0.3 = water stress, health drops | Irrigate proactively, not reactively |
| **Soil nutrients** | Below 0.3 = nutrient stress, health drops | Fertilize during vegetative/flowering |
| **Pest level** | Above 0.4 = pest damage, health drops | Apply pesticide when pests > 0.3 |
| **Weather** | Drought = -0.12 moisture, Storm = -0.08 health | Monitor forecast, prepare for drought |
| **Budget** | $500 total, actions cost $5-$40 | Don't overspend in early stages |
| **Harvest timing** | Must reach maturity (day 130+) with health > 0 | Keep crop alive through flowering |

---

## Agent Strategy Approaches

### Strategy 1: Conservative (Low Risk, Moderate Reward)

**Philosophy**: Maintain all metrics in safe zones, minimal spending.

```python
def conservative_agent(obs):
    """
    Conservative: Only act when metrics hit danger thresholds.
    Pros: Saves budget, simple
    Cons: May miss optimal timing, crop can die from slow damage
    """
    if obs.soil_moisture < 0.25:
        return CropAction(action_type="irrigate", intensity="low")
    if obs.pest_level > 0.5:
        return CropAction(action_type="apply_pesticide", intensity="low")
    if obs.soil_nutrients < 0.25:
        return CropAction(action_type="fertilize", intensity="low")
    if obs.growth_stage == "maturity" and obs.day >= 140:
        return CropAction(action_type="harvest", intensity="medium")
    return CropAction(action_type="wait", intensity="low")
```

**Expected**: May fail on tough seeds — waits too long, damage accumulates.

---

### Strategy 2: Proactive (Medium Risk, High Reward)

**Philosophy**: Stay ahead of problems. Higher thresholds, medium intensity.

```python
def proactive_agent(obs):
    """
    Proactive: Act before problems become critical.
    Pros: Keeps crop healthier, better harvest chance
    Cons: Higher spending, may waste resources
    """
    # Anticipate drought from forecast
    drought_coming = "drought" in obs.weather_forecast

    if obs.soil_moisture < 0.35 or (drought_coming and obs.soil_moisture < 0.5):
        return CropAction(action_type="irrigate", intensity="medium")
    if obs.pest_level > 0.3:
        return CropAction(action_type="apply_pesticide", intensity="medium")
    if obs.soil_nutrients < 0.4 and obs.growth_stage in ("seedling", "vegetative", "flowering"):
        return CropAction(action_type="fertilize", intensity="medium")
    if obs.growth_stage == "maturity" and obs.crop_health > 0.3 and obs.day >= 135:
        return CropAction(action_type="harvest", intensity="medium")
    return CropAction(action_type="wait", intensity="low")
```

**Expected**: Higher success rate, best balance of cost vs. outcome.

---

### Strategy 3: Aggressive (High Risk, Variable Reward)

**Philosophy**: Maximum investment for maximum yield.

```python
def aggressive_agent(obs):
    """
    Aggressive: Heavy resource use for maximum crop health.
    Pros: Best possible yield when it works
    Cons: Can run out of budget, wasteful spending
    """
    if obs.soil_moisture < 0.4:
        return CropAction(action_type="irrigate", intensity="high")
    if obs.pest_level > 0.2:
        return CropAction(action_type="apply_pesticide", intensity="high")
    if obs.soil_nutrients < 0.5:
        return CropAction(action_type="fertilize", intensity="high")
    if obs.growth_stage == "maturity" and obs.day >= 132:
        return CropAction(action_type="harvest", intensity="medium")
    return CropAction(action_type="wait", intensity="low")
```

**Expected**: Often runs out of budget. May get penalized for wasteful actions.

---

### Strategy 4: Weather-Adaptive (Smart Risk, Highest Reward)

**Philosophy**: Use weather forecast to make optimal decisions.

```python
def weather_adaptive_agent(obs):
    """
    Weather-Adaptive: Uses forecast to optimize timing.
    Pros: Most efficient resource usage, best long-term outcomes
    Cons: More complex logic, relies on forecast accuracy
    """
    forecast = obs.weather_forecast
    rain_coming = "rainy" in forecast or "storm" in forecast
    drought_coming = "drought" in forecast

    # Don't irrigate if rain is coming (save money)
    if obs.soil_moisture < 0.3 and not rain_coming:
        return CropAction(action_type="irrigate", intensity="medium")
    elif obs.soil_moisture < 0.2:  # Emergency irrigation even with rain forecast
        return CropAction(action_type="irrigate", intensity="high")

    # Irrigate preemptively before drought
    if drought_coming and obs.soil_moisture < 0.5:
        return CropAction(action_type="irrigate", intensity="medium")

    # Pest management
    if obs.pest_level > 0.35:
        intensity = "high" if obs.pest_level > 0.6 else "medium"
        return CropAction(action_type="apply_pesticide", intensity=intensity)

    # Nutrient management (stage-aware)
    if obs.soil_nutrients < 0.4 and obs.growth_stage in ("vegetative", "flowering"):
        return CropAction(action_type="fertilize", intensity="medium")
    elif obs.soil_nutrients < 0.3 and obs.growth_stage == "seedling":
        return CropAction(action_type="fertilize", intensity="low")

    # Harvest at optimal time
    if obs.growth_stage == "maturity" and obs.crop_health > 0.4 and obs.day >= 135:
        return CropAction(action_type="harvest", intensity="medium")

    return CropAction(action_type="wait", intensity="low")
```

**Expected**: Best overall performance — uses forecast to avoid waste.

---

## Test Scenarios

### Scenario 1: Normal Conditions

Tests basic agent competence with average weather.

```python
# Seeds that produce moderate weather
normal_seeds = [7, 15, 100, 42, 2026]
```

**What to verify**:
- Agent survives to maturity
- Positive total reward
- Budget not exhausted

### Scenario 2: Drought Stress

Tests agent's ability to handle water scarcity.

```python
# Run episodes and track moisture management
def test_drought_handling(agent_fn, seed):
    env = CropAdvisorEnvironment(seed=seed)
    obs = env.reset()
    drought_days = 0
    irrigations = 0
    while not env.done:
        if obs.weather_today == "drought":
            drought_days += 1
        action = agent_fn(obs)
        if action.action_type == "irrigate":
            irrigations += 1
        obs = env.step(action)
    return drought_days, irrigations, env.state.harvested
```

**What to verify**:
- Agent irrigates during/before droughts
- Moisture stays above 0.2 most of the time

### Scenario 3: Pest Outbreak

Tests agent's response to severe pest events.

```python
def test_pest_response(agent_fn, seed):
    env = CropAdvisorEnvironment(seed=seed)
    obs = env.reset()
    max_pest = 0
    pesticide_uses = 0
    while not env.done:
        max_pest = max(max_pest, obs.pest_level)
        action = agent_fn(obs)
        if action.action_type == "apply_pesticide":
            pesticide_uses += 1
        obs = env.step(action)
    return max_pest, pesticide_uses, env.state.harvested
```

**What to verify**:
- Agent applies pesticide before pest_level exceeds 0.5
- Not over-applying (wasting budget and getting penalty)

### Scenario 4: Budget Management

Tests agent's ability to survive with limited resources.

```python
def test_budget_efficiency(agent_fn, seed):
    env = CropAdvisorEnvironment(seed=seed)
    obs = env.reset()
    while not env.done:
        obs = env.step(agent_fn(obs))
    return {
        "budget_remaining": obs.budget_remaining,
        "budget_pct": obs.budget_remaining / 500 * 100,
        "harvested": env.state.harvested,
        "reward": env.state.total_reward,
    }
```

**What to verify**:
- Budget doesn't reach $0
- Higher budget remaining → budget efficiency bonus

### Scenario 5: Harvest Timing

Tests whether agent harvests at the optimal moment.

```python
def test_harvest_timing(agent_fn, seed):
    env = CropAdvisorEnvironment(seed=seed)
    obs = env.reset()
    while not env.done:
        obs = env.step(agent_fn(obs))
    return {
        "harvested": env.state.harvested,
        "harvest_day": obs.day if env.state.harvested else None,
        "health_at_harvest": obs.crop_health,
        "yield": obs.yield_estimate,
    }
```

**What to verify**:
- Harvest happens at maturity stage (day 130+)
- crop_health > 0.5 at harvest for good yield
- Never premature harvest (massive penalty)

---

## Running Tests

### Run All Strategy Comparisons

```bash
cd openenvHack
python tests/test_all_strategies.py
```

### Quick Single Test

```bash
python -c "
import sys; sys.path.insert(0, '.')
from crop_advisor_env.models import CropAction
from crop_advisor_env.server.crop_environment import CropAdvisorEnvironment

env = CropAdvisorEnvironment(seed=7)
obs = env.reset()
while not env.done:
    # Your strategy here
    obs = env.step(CropAction(action_type='wait', intensity='low'))
print(f'Reward: {env.state.total_reward:.2f}')
"
```

---

## Expected Outcomes

### Strategy Comparison — Actual Results (10 seeds each)

| Strategy | Harvest Rate | Avg Reward | Best Reward | Best Yield | Notes |
|----------|:-----------:|:----------:|:-----------:|:----------:|-------|
| Conservative | **0/10 (0%)** | 21.21 | 84.30 | 0.00 | Too passive, crop always dies |
| Proactive | **2/10 (20%)** | 51.67 | 246.27 | 4.43 | Good balance of cost/outcome |
| Aggressive | **1/10 (10%)** | 34.07 | 188.73 | 2.80 | Overspends, runs out of budget |
| Weather-Adaptive | **2/10 (20%)** | **57.26** | **254.48** | **4.76** | Best overall performance |

### Key Insights

1. **The environment rewards proactive management** — waiting too long is punished
2. **Weather forecast usage** gives a significant edge (20% forecast uncertainty is manageable)
3. **Budget efficiency matters** — overspending early means no resources for emergencies
4. **Pest management is critical** — the #1 cause of crop death is accumulated pest damage
5. **Nutrient timing is key** — fertilizing at vegetative/flowering stage gives the best reward

### What This Teaches an RL Agent

An RL agent training on this environment would learn:
- **Temporal planning**: Actions taken early affect outcomes 100+ days later
- **Resource allocation**: Limited budget requires efficient spending
- **Risk management**: Weather uncertainty requires contingency planning
- **Multi-objective optimization**: Balance yield, cost, and sustainability
