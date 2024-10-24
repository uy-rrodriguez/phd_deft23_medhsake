#!/bin/bash -l
#
#Helper functions for Bash scripts
#

read_config() {
    # Returns a configuration value found in the specially formatted config
    # files for slurm_run.sh and slurm_finetune.sh.
    #
    # Parameters:
    #   1) Path to the configuration file
    #   2) Task ID, identifier of the row to read
    #   3) Column number of the value to return
    #
    # Example:
    #   # Will return the value of column 2, in the row matching task id 45
    #   read_config "slurm_run_config.txt" 45 2

    CONFIG_FILE=$1
    TASK_ID=$2
    SEARCH_EXP="\$1==ArrayTaskID {print \$$3}"
    awk -v ArrayTaskID=$TASK_ID "$SEARCH_EXP" "$CONFIG_FILE"
}


run_with_time_track() {
    # Wrap execution of any command with a counter to know how long it takes.
    #
    # Examples:
    #   run_with_time_track echo "Test"
    #   run_with_time_track \
    #       echo \
    #           "Test" \
    #           2>&1 \
    #           | tee test.txt
    #

    echo "Start $(date +'%d/%m/%Y %H:%M:%S')"
    echo
    SECONDS=0

    $@

    DURATION=$SECONDS
    HOURS=$((DURATION / 60 / 60))
    MINS=$(((DURATION / 60) % 60))
    SECS=$((DURATION % 60))
    echo
    echo "End $(date +'%d/%m/%Y %H:%M:%S') (${HOURS}h ${MINS}m ${SECS}s)"
}
