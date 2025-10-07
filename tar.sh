PWD=$(basename $(pwd))
tar -czhvf "$PWD.tar.gz" \
    --exclude=venv \
    --exclude=__pycache__ \
    --exclude=.git \
    --exclude=models \
    --exclude=train_results \
    --exclude=wandb \
    --exclude=bkp_benoit \
    --exclude=output/analysis/forests \
    --exclude="$PWD.tar.gz" \
    .
