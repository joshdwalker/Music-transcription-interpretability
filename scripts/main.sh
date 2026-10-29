#!/usr/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --account=yunglu
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=6G

module load conda
conda activate .venv

cd /scratch/gilbreth/walk1098/Music-transcription-interpretability

mkdir huggingface
mkdir huggingface/hub
export HF_HUB_CACHE=/scratch/gilbreth/walk1098/Music-transcription-interpretability/huggingface/hub

pip install -r requirements.txt

python ./scripts/collect_activations.py