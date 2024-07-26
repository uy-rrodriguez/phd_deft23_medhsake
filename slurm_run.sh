#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --time=3:30:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#--SBATCH --array=1,3-24%3
#--SBATCH --array=4,12,16,18
#SBATCH --array=18
#

# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Static parameters
ENV=deft2023

# Run config file
CONFIG=slurm_run_config.txt

# Use fine-tuned model
USE_FINETUNED=1

# Extract the config for the current Slurm task
PROMPT_TPL=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $2}' $CONFIG)
NUM_SHOTS=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $3}' $CONFIG)
NUM_RUN=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $4}' $CONFIG)


echo "Activating conda environment $ENV"
conda activate $ENV

# Generate output directory and suffix for file names
SUFF=_shots${NUM_SHOTS}_medshake
if [ $USE_FINETUNED == 1 ]; then
    SUFF=${SUFF}_finetuned
fi
SUFF=${SUFF}_${NUM_RUN}

DIR=llama3/paper

# Select base or fine-tuned model
MODEL=meta-llama/Meta-Llama-3-8B
if [ $USE_FINETUNED == 1 ]; then
    MODEL=llama3_models/llama-3-8b-deft_001_20240719
fi


echo "Running LLaMa3 (shots $NUM_SHOTS, run $NUM_RUN)"
python run_llama3.py \
    output/$DIR/llama3-8b$SUFF.txt \
    data/dev-medshake-score.json \
    --model="$MODEL" \
    --template_id="'$PROMPT_TPL'" \
    --num_shots=$NUM_SHOTS \
    2>&1 \
    | tee logs/$DIR/llama3-8b$SUFF.txt
