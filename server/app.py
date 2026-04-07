"""
Server entry-point expected by OpenEnv multi-mode deployment.

Re-exports the FastAPI app from the crop_advisor_env package.
"""

from crop_advisor_env.server.app import app, main

__all__ = ["app", "main"]

if __name__ == "__main__":
    main()
