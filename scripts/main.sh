#!/usr/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --account=yunglu
#SBATCH --partition=a100-80GB
#SBATCH --qos=standby
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

module load conda
conda activate /home/walk1098/.conda/envs/2025.06-py313/.venv

cd /scratch/gilbreth/walk1098/Music-transcription-interpretability

python ./scripts/collect_activations.py