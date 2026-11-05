#!/bin/sh -l

#SBATCH -A cis240530-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=56G
#SBATCH --job-name collect-muscriptor-activations
#SBATCH -t 24:00:00  # 24 hours

module load conda
conda activate .venv

cd /anvil/scratch/x-jwalker5/Music-transcription-interpretability

mkdir huggingface
mkdir huggingface/hub
export HF_HUB_CACHE=/anvil/scratch/x-jwalker5/Music-transcription-interpretability/huggingface/hub

pip install -r requirements.txt

# Usage: python collect_actications.py /path/to/inputs /path/to/output_dir
#      where inputs is a text file containing absolute paths to every input .wav
python ./scripts/collect_activations.py slakh2100.txt activations

# Train the model
torchrun sparse_autoencoder/train.py