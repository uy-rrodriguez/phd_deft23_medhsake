#!/bin/bash -l
#
#SBATCH --job-name=Ric_BERT_Yanis_DEFT2023
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --constraint='GPURAM_Min_24GB'
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --mail-type=ALL
#

# Conda environment
ENV=deft2023


# Activate Conda environment before running the code
echo "Activating conda environment $ENV"
conda deactivate && conda activate $ENV


echo "Fine-tuning Bert model (Yanis code)"
python yanis/TrainFrenchMedMCQA-QA.py \
    --model_name "Dr-BERT/DrBERT-4GB" \
    2>&1 \
    | tee yanis/logs/$(date +"%Y%m%d_%H%M%S").txt
