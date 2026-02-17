#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EEG_CLEANER_PATH=$SCRIPT_DIR/../external/eeg_cleaner

if [ $# -ne 1 ]; then
    echo "Missing argument: path to database"
    exit -1
fi
export PYTHONPATH="${EEG_CLEANER_PATH}:${PYTHONPATH}"
python "${EEG_CLEANER_PATH}/scripts/2_clean_epochs.py" --path "${1}/derivatives/eeg_cleaner" --pca 10
