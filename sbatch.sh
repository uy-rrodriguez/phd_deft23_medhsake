#!/bin/bash -l
#

# Counts the number of output files generated for the given job id
function count_files() {
    echo $(ls -l slurm-$1*.out 2>/dev/null | wc -l)
}

# Handles closing of the tail command, to cancel the job and remove the output
function on_tail_close() {
    echo
    read -p "Cancel job? [y/N] " cancel
    if [[ "$cancel" == "y" ]]; then
        scancel $slurm_id
    fi

    read -p "Delete output file? [y/N] " del
    if [[ "$del" == "y" ]]; then
        rm "$slurm_file"
    fi
}


# Process arguments
SCRIPT=$1
if [[ "$1" == "run" ]]; then
    SCRIPT=slurm_run.sh
elif [[ "$1" == "finetune" ]]; then
    SCRIPT=slurm_finetune.sh
fi

info=$(sbatch "$SCRIPT")
# info="Submitted job id 903164"
echo $info

slurm_id=$(echo "$info" | grep -Po "\d+")

# Wait for output files to exist (job started)
num_files=$(count_files $slurm_id)
until [[ $num_files > 0 ]]; do
    num_files=$(count_files $slurm_id)
    sleep 1
done

if [[ $num_files == 1 ]]; then
    slurm_file=$(ls slurm-${slurm_id}*.out)
else
    echo Found multiple output files:
    ls -l slurm-${slurm_id}*.out

    read -p "Task number of the file to read: " slurm_task_id
    slurm_file=slurm-${slurm_id}_${slurm_task_id}.out
fi

# Capture Ctrl+C used to stop tail
trap on_tail_close INT

# Read Slurm output file
echo Reading file $slurm_file
echo
tail -f $slurm_file