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

# Collect the activations which the SAE trains over
# IMPORTANT: this only needs to be run ONCE ever; check if activations directory exists and is sufficiently full

# Gilbreth
# slakh2100.txt and slakh2100 should already be in the depot
sbatch main.sh

# Anvil
# Ensure you have slakh2100.txt and slakh2100 in this project directory
sbatch anvil_main.sh