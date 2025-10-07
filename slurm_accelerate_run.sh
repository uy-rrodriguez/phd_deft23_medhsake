#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=16G
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#SBATCH --array=38
#--SBATCH --array=61-72,50,51,53,54,56,57,59,60%3
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
DIR=llama3/accelerate
if [[ $MODEL_ID > 0 ]]; then
    MODEL=llama3_models/llama-3-8b-deft_${MODEL_NAME}
    echo "Using custom model $MODEL"

    SUFF=tuned_${MODEL_NAME}_

    # Sub-directory for logs and output files
    DIR="${DIR}/tuned_${MODEL_NAME}"
else
    DIR="${DIR}/$(date +'%Y%m%d')"
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


# Create accelerate launch command
GPUS_PER_NODE=4
HEAD_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
LAUNCHER="accelerate launch \
    --mixed_precision fp16 \
    --dynamo_backend no \
    --rdzv_backend c10d \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --main_process_ip $HEAD_NODE_IP \
    --main_process_port 29500 \
    "
SCRIPT="run_llama3_hf_accelerator.py"
SCRIPT_ARGS=" \
    --corpus_path=data/dev-medshake-score.json \
    --result_path=output/$DIR/llama3-8b_${SUFF}.txt \
    --model_path='$MODEL' \
    --prompt_template_id='$PROMPT_ID' \
    --num_shots=$NUM_SHOTS \
    --shots_full_answer=$WITH_ANSWER_TXT \
    "
CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

# This step is necessary because accelerate launch does not handle multiline
# arguments properly
echo "Accelerate Launch LLaMa3 (shots $NUM_SHOTS, run $NUM_RUN)"
echo "$CMD 2>&1 | tee logs/$DIR/llama3-8b_${SUFF}.txt"
srun $CMD 2>&1 | tee logs/$DIR/llama3-8b_${SUFF}.txt
