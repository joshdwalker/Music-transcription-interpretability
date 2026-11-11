#!/bin/sh -l

#SBATCH -A cis240530-gpu
#SBATCH -p gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export MASTER_ADDR MASTER_PORT

torchrun --nnodes="$SLURM_NNODES" --nproc-per-node=4 --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" sparse_autoencoder/train.py