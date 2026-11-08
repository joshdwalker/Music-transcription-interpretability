#!/bin/sh -l

#SBATCH -A cis240530-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --job-name train-muscriptor-sae
#SBATCH -t 24:00:00  # 24 hours

module load conda
conda activate .venv

cd /anvil/scratch/x-jwalker5/Music-transcription-interpretability

mkdir huggingface
mkdir huggingface/hub
export HF_HUB_CACHE=/anvil/scratch/x-jwalker5/Music-transcription-interpretability/huggingface/hub

pip install -r requirements.txt

# Train the model
torchrun sparse_autoencoder/train.py