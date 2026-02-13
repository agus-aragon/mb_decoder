#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EEG_CLEANER_PATH=$SCRIPT_DIR/../external/eeg_cleaner

if [ $# -ne 1 ]; then
    echo "Missing argument: path to databse"
    exit -1
fi

python "${EEG_CLEANER_PATH}/scripts/scripts/3_clean_ica.py" --path "${1}/derivatives/eeg_cleaner" --apply