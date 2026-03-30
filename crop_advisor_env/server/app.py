# Copyright (c) 2026 CropAdvisor RL Environment
# BSD 3-Clause License

"""
FastAPI application for the CropAdvisor RL environment.

Uses OpenEnv's create_fastapi_app helper to expose the environment
via standard HTTP/WebSocket endpoints.
"""

from openenv.core.env_server import create_fastapi_app

from ..models import CropAction, CropObservation
from .crop_environment import CropAdvisorEnvironment

# Create the FastAPI app using OpenEnv's helper
# Note: create_fastapi_app expects the Environment CLASS, not an instance
app = create_fastapi_app(CropAdvisorEnvironment, CropAction, CropObservation)
