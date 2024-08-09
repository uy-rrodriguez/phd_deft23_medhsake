#!/bin/bash -l
#
#SBATCH --job-name=Ric_DEFT2023_finetune
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --constraint='GPURAM_Min_16GB'
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#--SBATCH --array=2-3%1
#SBATCH --array=4
#

# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Whether to train on completions
# if [[ $TASK == 3 ]]
# then
#     TRAIN_COMPLETIONS=False
# else
#     TRAIN_COMPLETIONS=True
# fi
TRAIN_COMPLETIONS=True

# Static parameters
ENV=deft2023

# Run config file
# CONFIG=slurm_config.txt

# Extract the config for the current Slurm task
# PROMPT_TPL=$(awk -v ArrayTaskID=$TASK '$1==ArrayTaskID {print $2}' $CONFIG)


echo "Activating conda environment $ENV"
conda activate $ENV

# Generate suffix for file names
SUFF=00${TASK}_$(date +"%Y%m%d")


echo "Training LLaMa3"
python finetune_llama3.py \
    --train_dataset_name="data/train.json" \
    --new_model_path="llama3_models/llama-3-8b-deft_$SUFF/" \
    --run_name="llama-3-8b-deft_$SUFF" \
    --output_dir="train_results/$SUFF/" \
    --train_on_completions_only=$TRAIN_COMPLETIONS \
    2>&1 \
    | tee logs/llama3/finetuning/llama3-8b-deft_$SUFF.txt
