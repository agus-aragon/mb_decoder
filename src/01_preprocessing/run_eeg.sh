#!/bin/bash

# Run eeg_preproc_fmri.m for multiple subjects and tasks
# Usage: ./run_eeg.sh --subject 001 002 --task ES rest

# ===== CONFIGURATION =====
MATLAB_SCRIPT="eeg_prepro_fmri.m"
SCRIPT_DIR="/home/agusaragon/dev/mb_decoder/src/01_preprocessing"
DATA_PATH="/data/project/mb_decoder/data/bids/mb_decoder/"
EEGLAB_PATH="/home/agusaragon/dev/eeglab"
LOG_DIR="${DATA_PATH}derivatives/logs"
# =========================

# Parse command-line arguments
SUBJECTS=()
TASKS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --subject|-s)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SUBJECTS+=("$1")
                shift
            done
            ;;
        --task|-t)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                TASKS+=("$1")
                shift
            done
            ;;
        --data-path|-d)
            DATA_PATH="$2"
            shift 2
            ;;
        --eeglab-path|-e)
            EEGLAB_PATH="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --subject SUBJECTS... --task TASKS... [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --subject, -s SUBJECTS...    Subject IDs (e.g., 001 002)"
            echo "  --task, -t TASKS...          Task names (e.g., ES rest)"
            echo ""
            echo "Optional:"
            echo "  --data-path, -d PATH         BIDS data path"
            echo "  --eeglab-path, -e PATH       EEGLAB installation path"
            echo ""
            echo "Default paths:"
            echo "  Data: /data/project/mb_decoder/data/bids/mb_decoder/"
            echo "  EEGLAB: /home/agusaragon/dev/eeglab"
            echo ""
            echo "Examples:"
            echo "  $0 --subject 001 002 --task ES rest"
            echo "  $0 -s 001 -t ES -d /custom/data/ -e /custom/eeglab/"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ${#SUBJECTS[@]} -eq 0 ] || [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: Must specify at least one subject and one task"
    echo "Usage: $0 --subject SUBJECTS... --task TASKS..."
    exit 1
fi

# Ensure paths end with /
[[ "$DATA_PATH" != */ ]] && DATA_PATH="${DATA_PATH}/"
[[ "$EEGLAB_PATH" != */ ]] && EEGLAB_PATH="${EEGLAB_PATH}/"

# Validate paths exist
if [ ! -d "$DATA_PATH" ]; then
    echo "Error: Data path does not exist: $DATA_PATH"
    exit 1
fi

if [ ! -d "$EEGLAB_PATH" ]; then
    echo "Error: EEGLAB path does not exist: $EEGLAB_PATH"
    exit 1
fi

# Print configuration
echo "========================================="
echo "fMRI Artifact Removal - Batch Processing"
echo "========================================="
echo "Data path:   $DATA_PATH"
echo "EEGLAB path: $EEGLAB_PATH"
echo "Script dir:  $SCRIPT_DIR"
echo "Subjects:    ${SUBJECTS[*]}"
echo "Tasks:       ${TASKS[*]}"
echo "Combinations: $((${#SUBJECTS[@]} * ${#TASKS[@]}))"
echo "========================================="
echo ""

# Create logs directory
mkdir -p "$LOG_DIR"
echo "Logs will be saved to: $LOG_DIR"
echo ""

# Run all combinations
COUNTER=0
TOTAL=$((${#SUBJECTS[@]} * ${#TASKS[@]}))
FAILED=0
START_TIME=$(date +%s)

for subject in "${SUBJECTS[@]}"; do
    for task in "${TASKS[@]}"; do
        COUNTER=$((COUNTER + 1))
        
        echo ""
        echo "========================================="
        echo "[$COUNTER/$TOTAL] Subject: $subject, Task: $task"
        echo "========================================="
        
        # Check if input file exists before running
        INPUT_FILE="${DATA_PATH}sub-${subject}/eeg/sub-${subject}_task-${task}_eeg.vhdr"
        if [ ! -f "$INPUT_FILE" ]; then
            echo "WARNING: Input file not found, skipping: $INPUT_FILE"
            FAILED=$((FAILED + 1))
            continue
        fi
        
        # Run MATLAB with all paths as variables
        MATLAB_SCRIPT_FULL="$SCRIPT_DIR/$MATLAB_SCRIPT"
        matlab -nodisplay -nosplash -nodesktop -r \
            "cd('$SCRIPT_DIR'); data_path='$DATA_PATH'; eeglab_path='$EEGLAB_PATH'; subject_arg='$subject'; task_arg='$task'; run('$MATLAB_SCRIPT_FULL'); exit" 2>&1
        
        if [ $? -ne 0 ]; then
            echo "ERROR: Processing failed for subject $subject, task $task"
            FAILED=$((FAILED + 1))
        fi
    done
done

# Calculate total time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

# Final summary
echo ""
echo "========================================="
echo "Batch Processing Complete"
echo "========================================="
echo "Total processed: $COUNTER"
echo "Successful:      $((COUNTER - FAILED))"
echo "Failed:          $FAILED"
echo "Total time:      ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "Logs location:   $LOG_DIR"
echo "========================================="

if [ $FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi