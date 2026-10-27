#!/bin/sh -l

#SBATCH -A cis240530-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=48G
#SBATCH --job-name collect-muscriptor-activations
#SBATCH -t 01:00:00  # 1 hour

module load cuda conda
conda activate .venv

cd /anvil/scratch/x-jwalker5/Music-transcription-interpretability

pip install -r requirements.txt

python ./scripts/collect_activations.py