#!/bin/bash -l
#

BASES="logs output"
DIR=llama3/tuned_007_prompt2_spad

DEBUG=0

for b in $BASES; do
    for f in $b/$DIR/*.txt; do
        dest="${f/_nointro/}"
        if [ $DEBUG -eq 1 ]; then
            echo git mv "$f" "$dest"
            echo mv "$f" "$dest"
            exit
        else
            git mv "$f" "$dest" 2>/dev/null
            if [ $? -gt 0 ]; then
                mv "$f" "$dest"
            fi;
        fi;
    done
done
