#!/bin/sh -l

module load conda
conda activate .venv

cd /anvil/scratch/x-jwalker5/Music-transcription-interpretability

# Get number of pt files generated so far, and the total number needed to generate
num_activations=$(find activations -type f -name "*.pt" | wc -l)
total_activations=$(wc -l < slakh2100.txt)

# Set number of jobs to whatever is passed in as the first positional argument (or 5 by default)
if [ "$#" -ge 1 ]; then
    num_jobs=$1
else
    num_jobs=5
fi

subdivision_size=$(( (total_activations - num_activations + num_jobs) / num_jobs))

mkdir huggingface
mkdir huggingface/hub
export HF_HUB_CACHE=/anvil/scratch/x-jwalker5/Music-transcription-interpretability/huggingface/hub

pip install -r requirements.txt

# Submit some number of jobs each tasked with their own subset of slakh2100

i=0

while [ "$i" -lt "$num_jobs" ]; do
    # Usage: python collect_actications.py /path/to/inputs /path/to/output_dir
    #      where inputs is a text file containing absolute paths to every input .wav
    #      and start_index is the index of slakh2100 to start collecting activations over (default 0)
    #      and stop_index is the index to break at (default -1)
    sbatch ./scripts/anvil/collect_activations.sh $((num_activations + i * subdivision_size)) $((num_activations + (i + 1) * subdivision_size))
    i=$((i + 1))
done