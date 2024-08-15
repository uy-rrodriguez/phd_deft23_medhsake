#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#SBATCH --array=37-48%3
#--SBATCH --array=3,4,9,10,15,16,21,22%3
#--SBATCH --array=25-36%1
#--SBATCH --array=12
#

# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:="0"}

# Conda environment
ENV=deft2023

# Run config file
CONFIG=slurm_run_config.txt

# Extract the config for the current Slurm task
TASK_ID=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $1}' $CONFIG)
PROMPT_TPL=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $2}' $CONFIG)
NUM_SHOTS=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $3}' $CONFIG)
WITH_ANSWER_TXT=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $4}' $CONFIG)
MODEL_NAME=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $5}' $CONFIG)
NUM_RUN=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $6}' $CONFIG)

if [[ $TASK != $TASK_ID ]]
then
    >&2 echo "Error loading configuration for Task ID $TASK"
    exit 1
fi

# Handle Model ID to select model (0 = base model, 1+ = finetuned)
MODEL=meta-llama/Meta-Llama-3-8B
MODEL_ID=$(( $(echo "$MODEL_NAME" | grep -Po "\K^\d{3}") ))
if [[ $MODEL_ID > 0 ]]; then
    MODEL=llama3_models/llama-3-8b-deft_${MODEL_NAME}
    echo "Using custom model $MODEL"

    SUFF=tuned_${MODEL_NAME}_
fi

# Generate suffix for file names
SUFF=${SUFF}prompt${PROMPT_TPL}

# Suffix for runs without intro before few-shots
if [[ $PROMPT_TPL > 1 ]]; then
    SUFF=${SUFF}_nointro
fi

SUFF=${SUFF}_shots${NUM_SHOTS}

# Suffix for shots including answer text
if [[ $WITH_ANSWER_TXT == 1 ]]; then
    SUFF=${SUFF}_answertxt
fi

SUFF=${SUFF}_${NUM_RUN}

# Sub-directory for logs and output files
DIR=llama3/tuned_003_20240730

# Select appropriate prompt id in deft.py
if [[ $PROMPT_TPL == 1 || $PROMPT_TPL == 2 ]]; then
    PROMPT_ID=0
elif [[ $PROMPT_TPL == 3 ]]; then
    PROMPT_ID=1
fi


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p output/$DIR logs/$DIR

echo "Running LLaMa3 (shots $NUM_SHOTS, run $NUM_RUN)"
python run_llama3.py \
    output/$DIR/llama3-8b_${SUFF}.txt \
    data/dev-medshake-score.json \
    --model="$MODEL" \
    --prompt_template_id="'$PROMPT_ID'" \
    --num_shots=$NUM_SHOTS \
    --shots_full_answer=$WITH_ANSWER_TXT \
    2>&1 \
    | tee logs/$DIR/llama3-8b_${SUFF}.txt
