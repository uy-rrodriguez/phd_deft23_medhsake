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


run_with_emissions_track() {
    # Wrap execution of any command with a set of logging information to
    # calculate CO2 emissions.
    #
    # Examples:
    #   run_with_emissions_track echo "Test"
    #   run_with_emissions_track \
    #       echo \
    #           "Test" \
    #           2>&1 \
    #           | tee test.txt
    #

    function display_info() {
        # Display information from scontrol
        JOB_FIELDS="ArrayTaskId JobName RunTime StartTime EndTime NodeList \
                    NumNodes MinCPUsNode MinMemoryNode Features"
        GREP_REGEX="JobId"
        for f in $JOB_FIELDS; do
            GREP_REGEX="$GREP_REGEX|$f"
        done
        GREP_REGEX="(^| )($GREP_REGEX)=[^\s]*"

        echo
        echo "Job information:"
        echo "────────────────"
        scontrol show job $SLURM_JOB_ID | grep -Po "$GREP_REGEX" | grep -Po "\w.*"
        echo "────────────────"
    }

    trap display_info INT TERM


    # Node allocated, necessary to then calculate CO2 emissions
    # TODO: Add support for multiple nodes
    NODE=$(scontrol show job $SLURM_JOB_ID 2>/dev/null \
           | grep -Po " NodeList=[^\s]+" | grep -Po "=.+" | grep -Po "\w+")
    GPU=$(sinfo -o "%N %G" | grep "${NODE}" | grep -Po "gpu:.+" \
          | grep -Po ":.+?:" | grep -Po "[^:]+")
    echo "Allocated nodes: ${NODE} (${GPU})"
    echo

    $@

    display_info
}
