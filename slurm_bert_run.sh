#!/bin/bash -l
#
#SBATCH --job-name=Ric_BERT_DEFT2023
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#SBATCH --array=1-2
#

source functions.sh


# Handle Slurm Task ID
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Conda environment
ENV=deft2023

# Run parameters
DEBUG=0
NUM_RUN=0

# Parameters of BERT model
MODEL_FAMILY=drbert
MODEL_NAME=DrBERT-4GB
MODEL_NAME_LC=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
MODEL_URL=Dr-BERT/$MODEL_NAME
# MODEL_LOCAL_ID=$(printf "%03d" $TASK)
# MODEL_LOCAL_VER=${MODEL_LOCAL_ID}_20240912
if [[ $TASK == 1 ]]; then
    MODEL_LOCAL_VER=005_20240924_1252
elif [[ $TASK == 2 ]]; then
    MODEL_LOCAL_VER=005_20240924_1412
fi
MODEL_LOCAL=models/$MODEL_FAMILY/$MODEL_NAME_LC-deft_$MODEL_LOCAL_VER

MODEL=$MODEL_LOCAL

# Output parameters
DIR=$MODEL_FAMILY/tuned_$MODEL_LOCAL_VER
FILENAME=$MODEL_NAME_LC
SUFF=$NUM_RUN


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p output/$DIR logs/$DIR

echo "Running inference with Bert model"
# python run_bert_mcq.py \
run_with_time_track \
    python run_bert_mcq.py \
        --method_name=run_inference \
        --model-path="$MODEL" \
        --corpus-path="data/dev-medshake-score.json" \
        --result-path="output/$DIR/${FILENAME}_${SUFF}.txt" \
        --debug=$DEBUG \
        2>&1 \
        | tee logs/$DIR/${FILENAME}_${SUFF}.txt
