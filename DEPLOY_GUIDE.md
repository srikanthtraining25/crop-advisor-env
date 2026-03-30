# Deploying CropAdvisor to Hugging Face Spaces

## Prerequisites

1. **Hugging Face Account**: Sign up at [huggingface.co](https://huggingface.co)
2. **HF CLI**: Install with `pip install huggingface_hub`
3. **HF Token**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. **Git LFS**: `git lfs install`
5. **Docker Desktop**: Must be running for local testing

## Step 1: Login to Hugging Face

```bash
huggingface-cli login
# Paste your HF token when prompted
```

## Step 2: Create a New Space

```bash
# Create Space (Docker SDK)
huggingface-cli repo create crop-advisor-env --type space --space-sdk docker
```

Or create manually at: https://huggingface.co/new-space
- Space name: `crop-advisor-env`
- SDK: `Docker`
- License: `BSD-3-Clause`

## Step 3: Clone and Push

```bash
# Clone the empty Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/crop-advisor-env
cd crop-advisor-env

# Copy our files into the Space
mkdir crop_advisor_env
cp -r ../openenvHack/crop_advisor_env/* ./crop_advisor_env/
cp ../openenvHack/requirements.txt ./requirements.txt
cp ../openenvHack/Dockerfile ./Dockerfile
cp ../openenvHack/HF_README.md ./README.md

# Commit and push
git add .
git commit -m "Add CropAdvisor RL Environment"
git push
```

## Step 4: Verify Deployment

Once pushed, HF Spaces will build the Docker image automatically.

```python
# Test the deployed environment
from crop_advisor_env import CropAction, CropAdvisorEnv

with CropAdvisorEnv(base_url="https://YOUR_USERNAME-crop-advisor-env.hf.space").sync() as env:
    obs = env.reset()
    print(obs.message)
    obs = env.step(CropAction(action_type="irrigate", intensity="medium"))
    print(obs.message)
```

## Step 5: Local Docker Test (Before Push)

```bash
# Start Docker Desktop first!
cd openenvHack
docker build -f Dockerfile -t crop-advisor-env .
docker run -p 8000:8000 crop-advisor-env

# In another terminal:
curl  
```

## Alternative: Using openenv push

If the `openenv` CLI is available:
```bash
cd openenvHack
openenv push crop_advisor_env
```

This auto-creates the HF Space and pushes the environment.
