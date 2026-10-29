# Music-transcription-interpretability
Right now, simply probe AMT with a SAE

# Setting up the virtual environment
module load conda
conda create -n .venv python=3.11

# Requirements
pip install -r requirements.txt

# Muscriptor license
1. Accept the model license at https://huggingface.co/MuScriptor/muscriptor-large
     (access is granted automatically).
2. Authenticate on this machine:
     set the HF_TOKEN environment variable to a read token
          from https://huggingface.co/settings/tokens
     hf auth login --token <your_token>