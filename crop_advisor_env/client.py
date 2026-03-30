# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
CropAdvisor Environment Client.

Provides a typed EnvClient for interacting with the CropAdvisor
environment, both locally and remotely (via Hugging Face Spaces).
"""

from openenv.core.env_client import EnvClient

from .models import CropAction, CropObservation, CropState


class CropAdvisorEnv(EnvClient):
    """
    Typed client for the CropAdvisor RL environment.

    Usage (async):
        async with CropAdvisorEnv(base_url="https://...") as env:
            obs = await env.reset()
            obs = await env.step(CropAction(action_type="irrigate", intensity="medium"))

    Usage (sync):
        with CropAdvisorEnv(base_url="https://...").sync() as env:
            obs = env.reset()
            obs = env.step(CropAction(action_type="irrigate", intensity="medium"))
    """

    # Type hints for the OpenEnv framework
    action_type = CropAction
    observation_type = CropObservation
    state_type = CropState

    def _step_payload(self, action: CropAction) -> dict:
        """Convert action into payload for HTTP request."""
        return action.model_dump()

    def _parse_result(self, data: dict) -> CropObservation:
        """Parse step/reset result into CropObservation."""
        obs_data = data.get("observation", data)
        return CropObservation(**obs_data)

    def _parse_state(self, data: dict) -> CropState:
        """Parse state result into CropState."""
        state_data = data.get("state", data)
        return CropState(**state_data)
