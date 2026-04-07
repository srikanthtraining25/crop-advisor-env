"""
Server entry-point expected by OpenEnv multi-mode deployment.
"""

import uvicorn
from crop_advisor_env.server.app import app


def main():
    """Start the CropAdvisor environment server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
