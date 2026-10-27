#!/bin/sh -l

#SBATCH -A x-jwalker5
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --job-name collect-muscriptor-activations
#SBATCH -t 01:00:00  # 1 hour

module load conda
conda activate .venv

cd /anvil/scratch/x-jwalker5/Music-transcription-interpretability

pip install -r requirements.txt

python ./scripts/collect_activations.py