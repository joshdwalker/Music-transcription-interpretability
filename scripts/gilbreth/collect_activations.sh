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

# Usage: python collect_actications.py /path/to/inputs /path/to/output_dir start_index stop_index
#      where inputs is a text file containing absolute paths to every input .wav
#      and start_index is the index of slakh2100 to start collecting activations over (default 0)
#      and stop_index is the index to break at (default -1)
python ./scripts/collect_activations.py /depot/yunglu/data/transcription/slakh2100.txt activations