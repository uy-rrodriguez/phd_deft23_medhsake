#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_16GB&GPURAM_Max_24GB'
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#--SBATCH --array=402-448%2
#--SBATCH --array=450-495%2
#SBATCH --array=601-624%2
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
MODEL_REF=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $5}' $CONFIG)
NUM_RUN=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $6}' $CONFIG)

if [[ $TASK != $TASK_ID ]]
then
    >&2 echo "Error loading configuration for Task ID $TASK"
    exit 1
fi


# Handle model selection, based on configured name (family/id)
MODEL_FAMILY=$(echo "$MODEL_REF" | grep -Po "^[\w_]+/?" | grep -Po "[\w_]+") # Ref before /
MODEL_NAME=$(echo "$MODEL_REF" | grep -Po "/[\w_]+" | grep -Po "[\w_]+") # Ref after /
MODEL_ID=$(echo "$MODEL_REF" | grep -Po "/[\d]+_" | grep -Po "[\d]+") # Numeric prefix after /

RUN_SCRIPT=run_llm.py

if [[ "$MODEL_FAMILY" == "llama3" ]]; then
MODEL=meta-llama/Meta-Llama-3-8B
    LOCAL_MODEL=models/llama3/llama-3-8b-deft_${MODEL_NAME}
    DIR=llama3
    FILENAME=llama3-8b
    RUN_SCRIPT=run_llama3.py

elif [[ "$MODEL_FAMILY" == "mmed_llama3" ]]; then
    MODEL=Henrychur/MMed-Llama-3-8B
    LOCAL_MODEL=models/mmed-llama3/mmed-llama3-8b-deft_${MODEL_NAME}
    DIR=mmed-llama3
    FILENAME=mmed-llama3-8b

elif [[ "$MODEL_FAMILY" == "mistral" ]]; then
    MODEL=mistralai/Mistral-7B-v0.3
    LOCAL_MODEL=models/mistral/mistral-7b-deft_${MODEL_NAME}
    DIR=mistral
    FILENAME=mistral

elif [[ "$MODEL_FAMILY" == "biomistral" ]]; then
    MODEL=BioMistral/BioMistral-7B
    LOCAL_MODEL=models/biomistral/biomistral-7b-deft_${MODEL_NAME}
    DIR=biomistral
    FILENAME=biomistral

else
    >&2 echo "Model family not recognised: '$MODEL_FAMILY'"
    exit 2
fi

# Model ID: if empty use base model, otherwise use finetuned
if [[ "$MODEL_ID" != "" ]]; then
    MODEL=$LOCAL_MODEL
    echo "Using custom model $MODEL"

    SUFF=tuned_${MODEL_ID}_

    # Sub-directory for logs and output files
    DIR="${DIR}/tuned_${MODEL_NAME}"
else
    DIR="${DIR}/$(date +'%Y%m%d')"
fi

# Generate suffix for file names
SUFF=${SUFF}prompt${PROMPT_TPL}

# Suffix for runs without intro before few-shots
# if [[ $PROMPT_TPL > 1 ]]; then
#     SUFF=${SUFF}_nointro
# fi

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
elif [[ $PROMPT_TPL == 4 ]]; then
    # Prompt based on LLaMaInstructionsFrenchMedMCQA
    PROMPT_ID=2
fi


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p output/$DIR logs/$DIR

echo "Running inference with $MODEL_REF (shots $NUM_SHOTS, run $NUM_RUN)"
python $RUN_SCRIPT \
    --corpus_path=data/dev-medshake-score.json \
    --result_path=output/$DIR/${FILENAME}_${SUFF}.txt \
    --model_path="$MODEL" \
    --prompt_template_id="'$PROMPT_ID'" \
    --num_shots=$NUM_SHOTS \
    --shots_full_answer=$WITH_ANSWER_TXT \
    2>&1 \
    | tee logs/$DIR/${FILENAME}_${SUFF}.txt
