#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023_finetune
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --constraint='GPURAM_Min_32GB'
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#--SBATCH --array=2-3%1
#SBATCH --array=7
#

# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Static parameters
ENV=deft2023

# Run config file
# CONFIG=slurm_config.txt

# Extract the config for the current Slurm task
# PROMPT_TPL=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $2}' $CONFIG)

# Whether to train on completions
# if [[ $TASK == 3 ]]
# then
#     TRAIN_COMPLETIONS=False
# else
#     TRAIN_COMPLETIONS=True
# fi
TRAIN_COMPLETIONS=True

# Choice of prompt to use for fine-tuning
PROMPT_TPL=2

# Select appropriate prompt id in deft.py
if [[ $PROMPT_TPL == 1 || $PROMPT_TPL == 2 ]]; then
    PROMPT_ID=0
elif [[ $PROMPT_TPL == 3 ]]; then
    PROMPT_ID=1
elif [[ $PROMPT_TPL == 4 ]]; then
    # Prompt based on LLaMaInstructionsFrenchMedMCQA
    PROMPT_ID=2
fi

# Whether to include the full answer text in the responses
FULL_ANSWERS=True
if [[ $SLURM_ARRAY_TASK_ID == 51 ]]; then
    FULL_ANSWERS=False
fi

# Whether to use a special token for padding, instead of EOS
SPECIAL_PAD=False
if [[ $SLURM_ARRAY_TASK_ID == 6 || $SLURM_ARRAY_TASK_ID == 7 ]]; then
    SPECIAL_PAD=True
fi


echo "Activating conda environment $ENV"
conda activate $ENV

# Generate suffix for file names
SUFF=00${TASK}_$(date +"%Y%m%d")


echo "Training LLaMa3"
python finetune_llama3.py \
    --train_dataset_name="data/train.json" \
    --new_model_path="models/llama3/llama-3-8b-deft_$SUFF/" \
    --run_name="llama-3-8b-deft_$SUFF" \
    --output_dir="train_results/$SUFF/" \
    --train_on_completions_only=$TRAIN_COMPLETIONS \
    --use_special_pad_token=$SPECIAL_PAD \
    --prompt_template_id=$PROMPT_ID \
    --include_full_answers=$FULL_ANSWERS \
    --max_seq_length=512 \
    2>&1 \
    | tee logs/llama3/finetuning/llama3-8b-deft_$SUFF.txt
