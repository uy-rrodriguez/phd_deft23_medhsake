#!/bin/bash -l
#
#SBATCH --job-name=Ric_BERT_finetune_DEFT2023
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_24GB&GPURAM_Max_32GB'
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#
# Run multiple commands in parallel:
#SBATCH --array=7
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
TASK=${SLURM_ARRAY_TASK_ID:=0}

# Conda environment
ENV=deft2023

# Run parameters
DEBUG=0
EPOCHS=10
# REPORT_TO="'none'"
REPORT_TO="'wandb'"

# Run config file
CONFIG=slurm_bert_finetune_config.txt

# Extract the config for the current Slurm task
TASK_ID=$(              read_config $CONFIG $TASK 1)
ANSWER_ADD_PREFIX=$(    read_config $CONFIG $TASK 2)
CTX_INCLUDE_CHOICES=$(  read_config $CONFIG $TASK 3)
MODEL_REF=$(            read_config $CONFIG $TASK 4)
TASK_TYPE=$(            read_config $CONFIG $TASK 5)
TOTAL_RUNS=$(           read_config $CONFIG $TASK 6)

if [[ $TASK != $TASK_ID ]]
then
    >&2 echo "Error loading configuration for Task ID $TASK"
    exit 1
fi


# Handle model selection, based on configured name (family/id)
MODEL_FAMILY=$(echo "$MODEL_REF" | grep -Po "^[\w_]+/?" | grep -Po "[\w_]+") # Ref before /
# MODEL_NAME=$(echo "$MODEL_REF" | grep -Po "/[\w_]+" | grep -Po "[\w_]+") # Ref after /
# MODEL_ID=$(echo "$MODEL_REF" | grep -Po "/[\d]+_" | grep -Po "[\d]+") # Numeric prefix after /

if [[ "$MODEL_FAMILY" == "drbert" ]]; then
    MODEL_NAME=DrBERT-4GB
    MODEL_URL=Dr-BERT/$MODEL_NAME
    BATCH_SIZE=32

elif [[ "$MODEL_FAMILY" == "camembert" ]]; then
    MODEL_NAME=camembert-base
    MODEL_URL=almanach/$MODEL_NAME
    BATCH_SIZE=1

else
    >&2 echo "Model family not recognised: '$MODEL_FAMILY'"
    exit 2
fi

# Model parameters
MODEL_NAME_LC=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')
MODEL_LOCAL_ID=$(printf "%03d" $TASK)
MODEL_LOCAL_VER=${MODEL_LOCAL_ID}_$(date +"%Y%m%d")_$(date +"%H%M")
MODEL_LOCAL_VER_DATE_ONLY=${MODEL_LOCAL_ID}_$(date +"%Y%m%d")
MODEL_LOCAL_NAME=$MODEL_NAME_LC-deft_$MODEL_LOCAL_VER
BASE_MODEL_LOCAL_PATH=models/$MODEL_FAMILY/$MODEL_LOCAL_NAME

# Handle BERT task type
if [[ "$TASK_TYPE" == "mcq" ]]; then
    SCRIPT=run_bert_mcq.py
    LOGS_SUBDIR=mcq
elif [[ "$TASK_TYPE" == "seq" ]]; then
    SCRIPT=run_bert_classification.py
    LOGS_SUBDIR=classification
else
    >&2 echo "BERT task type not recognised: '$TASK_TYPE'"
    exit 3
fi


# Output parameters
BASE_RUN_NAME=$MODEL_LOCAL_NAME

LOGS_DIR=logs/$MODEL_FAMILY/$LOGS_SUBDIR/finetuning
TRAIN_OUT_DIR=train_results/$MODEL_FAMILY/$MODEL_LOCAL_VER

TEST_OUT_DIR=output/$MODEL_FAMILY/tuned_$(date +"%Y%m%d")
BASE_TEST_OUT_FILENAME=$MODEL_LOCAL_NAME


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda activate $ENV

# Create output directories
mkdir -p $LOGS_DIR $TRAIN_OUT_DIR $TEST_OUT_DIR

# Loop over TOTAL_RUNS
for NUM_RUN in $(seq 1 $TOTAL_RUNS); do
    RUN_NAME=$BASE_RUN_NAME
    TEST_OUT_FILENAME=$BASE_TEST_OUT_FILENAME
    MODEL_LOCAL_PATH=$BASE_MODEL_LOCAL_PATH

    if [[ $TOTAL_RUNS -gt 1 ]]; then
        RUN_NAME=${RUN_NAME}_$NUM_RUN
        TEST_OUT_FILENAME=${TEST_OUT_FILENAME}_$NUM_RUN
        MODEL_LOCAL_PATH=${MODEL_LOCAL_PATH}_$NUM_RUN
    fi

    if [[ $NUM_RUN -gt 1 ]]; then
        echo
        echo "============================================================"
        echo
    fi

    echo "Fine-tuning Bert model (run $NUM_RUN)"
    run_with_time_track \
        python $SCRIPT \
            --model-path="$MODEL_URL" \
            --train-corpus-path="data/train.json" \
            --dev-corpus-path="data/dev.json" \
            --train-run-name=$RUN_NAME \
            --epochs=$EPOCHS \
            --batch-size=$BATCH_SIZE \
            --report-to=$REPORT_TO \
            --new-model-path="$MODEL_LOCAL_PATH" \
            --train-output-dir="$TRAIN_OUT_DIR" \
            --ctx-include-choices=$CTX_INCLUDE_CHOICES \
            --answer-add-prefix=$ANSWER_ADD_PREFIX \
            --test-corpus-path="data/test-medshake-score.json" \
            --test-result-path="$TEST_OUT_DIR/$TEST_OUT_FILENAME.txt" \
            --debug=$DEBUG \
            2>&1 \
            | tee $LOGS_DIR/$RUN_NAME.txt
done
