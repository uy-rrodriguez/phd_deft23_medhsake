#!/bin/bash -l
#
#SBATCH --job-name=Ric_LLM_DEFT2023
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_16GB&GPURAM_Max_32GB'
#SBATCH --time=3:00:00
#SBATCH --requeue
#--SBATCH --mail-type=ALL
#SBATCH --mail-type=ARRAY_TASKS,FAIL,INVALID_DEPEND,REQUEUE,TIME_LIMIT
#
# Run multiple commands in parallel:
#--SBATCH --array=402-448%2
#--SBATCH --array=450-495%2
#SBATCH --array=601-624%2
#

source functions.sh


# Node allocated, necessary to then calculate CO2 emissions
# TODO: Add support for multiple nodes
NODE=$(scontrol show job $SLURM_JOB_ID 2>/dev/null \
       | grep -Po " NodeList=[^\s]+" | grep -Po "=.+" | grep -Po "\w+")
GPU=$(sinfo -o "%N %G" | grep "${NODE}" | grep -Po "gpu:.+" \
      | grep -Po ":.+?:" | grep -Po "[^:]+")
echo "Allocated nodes: ${NODE} (${GPU})"


# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:="0"}

# Conda environment
ENV=deft2023

# Run config file
CONFIG=slurm_llm_run_config.txt

# Extract the config for the current Slurm task
TASK_ID=$(      read_config $CONFIG $TASK 1)
PROMPT_TPL=$(   read_config $CONFIG $TASK 2)
NUM_SHOTS=$(    read_config $CONFIG $TASK 3)
ANSWER_TXT=$(   read_config $CONFIG $TASK 4)
MODEL_REF=$(    read_config $CONFIG $TASK 5)
NUM_RUN=$(      read_config $CONFIG $TASK 6)

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

elif [[ "$MODEL_FAMILY" == "llama3_70b" ]]; then
    MODEL=meta-llama/Meta-Llama-3-70B
    LOCAL_MODEL=models/llama3/llama-3-70b-deft_${MODEL_NAME}
    DIR=llama3-70b
    FILENAME=llama3-70b

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

elif [[ "$MODEL_FAMILY" == "apollo" ]]; then
    MODEL=FreedomIntelligence/Apollo-7B
    LOCAL_MODEL=models/apollo/apollo-7b-deft_${MODEL_NAME}
    DIR=apollo
    FILENAME=apollo-7b

else
    >&2 echo "Model family not recognised: '$MODEL_FAMILY'"
    exit 2
fi

# Model ID: if empty use base model, otherwise use finetuned
if [[ "$MODEL_ID" != "" ]]; then
    MODEL=$LOCAL_MODEL
    SUFF=tuned_${MODEL_ID}_
    # Sub-directory for logs and output files
    DIR="${DIR}/tuned_${MODEL_NAME}"
else
    DIR="${DIR}/base_$(date +'%Y%m%d')"
fi

# NOTE: Change output directory for tests
# DIR="test/${DIR}"

# Generate suffix for file names
SUFF=${SUFF}prompt${PROMPT_TPL}

# Suffix for runs without intro before few-shots
# if [[ $PROMPT_TPL > 1 ]]; then
#     SUFF=${SUFF}_nointro
# fi

SUFF=${SUFF}_shots${NUM_SHOTS}

# Suffix for shots including answer text
if [[ $ANSWER_TXT == 1 ]]; then
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

# Use of special padding token must be True if the fine-tuned model was trained
# with the same parameter
SPECIAL_PAD=False
# SPECIAL_PAD=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $7}' $CONFIG)
# if [[ $SPECIAL_PAD == 1 ]]; then
#     SPECIAL_PAD=True
# else
#     SPECIAL_PAD=False
# fi

# Parameters that affect how test is generated between runs
SAMPLING=None   # Whether to use sampling with a default value for Top-p
SAMPLING_TOP_P=None  # Top-p value to use with sampling
SAMPLING_TOP_K=None  # Top-k value to use with sampling
VARY_TEMP=false # Whether to change temperature between runs
TEMP=None       # Custom temperature
NUM_BEAMS=None  # Custom number of beams (Beam-search decoding) for inference

# if [[ ( "$MODEL_FAMILY" == *mistral || "$MODEL_FAMILY" == apollo ) && $NUM_SHOTS == 0 ]]; then
#     # VARY_TEMP=true
#     SAMPLING=true
#     SAMPLING_TOP_P=0.95
#     SAMPLING_TOP_K=0
# fi

# if $VARY_TEMP; then
#     TEMPS=("1.0" "0.7" "0.4" "0.1")
#     TEMP=${TEMPS[ $((NUM_RUN - 1)) ]}
#     echo "Using varying temperature: $TEMP for run $NUM_RUN"
# fi

# if [[ "$MODEL_FAMILY" == mistral && $NUM_SHOTS == 0 ]]; then
#     NUM_BEAMS=5
#     echo "Using custom beams: $NUM_BEAMS"
# fi


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p output/$DIR logs/$DIR

echo "Running inference with '$MODEL' (shots $NUM_SHOTS, run $NUM_RUN)"
        # --corpus_path=data/test.json \
run_with_time_track \
    python $RUN_SCRIPT \
        --corpus_path=data/test-medshake-score.json \
        --result_path=output/$DIR/${FILENAME}_${SUFF}.txt \
        --model_path="$MODEL" \
        --use_special_pad_token=$SPECIAL_PAD \
        --prompt_template_id="'$PROMPT_ID'" \
        --num_shots=$NUM_SHOTS \
        --shots_full_answer=$ANSWER_TXT \
        --do_sample=$SAMPLING \
        --top_p=$SAMPLING_TOP_P \
        --top_k=$SAMPLING_TOP_K \
        --temperature=$TEMP \
        --num_beams=$NUM_BEAMS \
        2>&1 \
        | tee logs/$DIR/${FILENAME}_${SUFF}.txt
