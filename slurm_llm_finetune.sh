#!/bin/bash -l
#
#SBATCH --job-name=Ric_LLM_finetune_DEFT2023
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --constraint='GPURAM_Min_32GB'
#--SBATCH --constraint='GPURAM_Min_80GB'
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#SBATCH --array=9-13%2
#

source functions.sh


# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Static parameters
ENV=deft2023

# Run parameters
# REPORT_TO="'none'"
REPORT_TO="'wandb'"

# Run config file
CONFIG=slurm_llm_finetune_config.txt

# Extract the config for the current Slurm task
TASK_ID=$(      read_config $CONFIG $TASK 1)
COMPLETIONS=$(  read_config $CONFIG $TASK 2)
SPECIAL_PAD=$(  read_config $CONFIG $TASK 3)
PROMPT_TPL=$(   read_config $CONFIG $TASK 4)
FULL_ANSWERS=$( read_config $CONFIG $TASK 5)
MAX_SEQ_LEN=$(  read_config $CONFIG $TASK 6)
MODEL_REF=$(    read_config $CONFIG $TASK 7)

if [[ $TASK != $TASK_ID ]]
then
    >&2 echo "Error loading configuration for Task ID $TASK"
    exit 1
fi

# Generate suffix for model name
SUFF=$(printf "%03d" $TASK)_$(date +"%Y%m%d")


# Handle model selection, based on configured name (family/id)
MODEL_FAMILY=$(echo "$MODEL_REF" | grep -Po "^[\w_]+/?" | grep -Po "[\w_]+") # Ref before /

if [[ "$MODEL_FAMILY" == "llama3" ]]; then
    MODEL=meta-llama/Meta-Llama-3-8B
    MODEL_NAME=llama-3-8b-deft_$SUFF

elif [[ "$MODEL_FAMILY" == "llama3_70b" ]]; then
    MODEL=meta-llama/Meta-Llama-3-70B
    MODEL_NAME=llama-3-70b-deft_$SUFF

elif [[ "$MODEL_FAMILY" == "mistral" ]]; then
    MODEL=mistralai/Mistral-7B-v0.3
    MODEL_NAME=mistral-7b-deft_$SUFF

elif [[ "$MODEL_FAMILY" == "biomistral" ]]; then
    MODEL=BioMistral/BioMistral-7B
    MODEL_NAME=biomistral-7b-deft_$SUFF

elif [[ "$MODEL_FAMILY" == "apollo" ]]; then
    MODEL=FreedomIntelligence/Apollo-7B
    MODEL_NAME=apollo-7b-deft_$SUFF

else
    >&2 echo "Model family not recognised: '$MODEL_FAMILY'"
    exit 2
fi


# Choice of prompt to use for fine-tuning
# Select appropriate prompt id in deft.py
if [[ $PROMPT_TPL == 1 || $PROMPT_TPL == 2 ]]; then
    PROMPT_ID=0
elif [[ $PROMPT_TPL == 3 ]]; then
    PROMPT_ID=1
elif [[ $PROMPT_TPL == 4 ]]; then
    # Prompt based on LLaMaInstructionsFrenchMedMCQA
    PROMPT_ID=2
fi


echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p logs/$MODEL_FAMILY/finetuning

echo "Training LLM ($MODEL_NAME)"
run_with_time_track \
    python finetune_llm.py \
        --model_checkpoint="$MODEL" \
        --new_model_path="models/$MODEL_FAMILY/$MODEL_NAME/" \
        --run_name=$MODEL_NAME \
        --train_dataset_name="data/train.json" \
        --output_dir="train_results/$MODEL_FAMILY/$MODEL_NAME/" \
        --train_on_completions_only=$COMPLETIONS \
        --use_special_pad_token=$SPECIAL_PAD \
        --prompt_template_id=$PROMPT_ID \
        --include_full_answers=$FULL_ANSWERS \
        --max_seq_length=$MAX_SEQ_LEN \
        --report-to=$REPORT_TO \
        2>&1 \
        | tee logs/$MODEL_FAMILY/finetuning/$MODEL_NAME.txt
