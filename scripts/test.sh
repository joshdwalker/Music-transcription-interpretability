#!/usr/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --account=csso
#SBATCH --partition=a10
#SBATCH --qos=standby
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=1G

module load cuda anaconda
conda activate /home/walk1098/.conda/envs/2025.06-py313/.venv

cd /scratch/gilbreth/walk1098/Music-transcription-interpretability

python ./scripts/test.py