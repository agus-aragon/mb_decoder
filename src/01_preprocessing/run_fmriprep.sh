#!/bin/bash

## ##  ############### Console usage: bash run_fmriprep.sh  ############### ## ##
# Run fMRIPrep for processing MRI data of all subjects                          #
# INPUT:                                                                        #
#   1. bids_root_dir: Path to BIDS dataset                                      #
#   2. output_dir: Derivative output directory                                  #
#   3. FS_LICENSE: FreeSurfer license path                                      #
#   4. WORK_DIR: Temporary working directory                                    #
#################################################################################

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mb_decoder
SUBJECT="$1"

export TEMPLATEFLOW_HOME="/data/project/mb_decoder/templateflow_cache"
echo "Initializing script"
export FREESURFER_HOME="/data/project/tools/juseless_tools/freesurfer_7.4.1"
source $FREESURFER_HOME/freesurfer.sh
bids_root_dir=/data/project/mb_decoder/data/bids/mb_decoder
output_dir="${bids_root_dir}/derivatives/fmriprep/"
export FSLOUTPUTTYPE=NIFTI
cp ~/freesurfer_license.txt /tmp/freesurfer_license.txt
export FSLDIR="/data/project/tools/juseless_tools/fsl_6.0.4-patched2"
source $FSLDIR/fsl.sh
export ANTSPATH="/data/project/tools/juseless_tools/ants_2.5.0"
source $ANTSPATH/ants.sh
export AFNIDIR="/data/project/tools/juseless_tools/afni_24.3.06/afni"
source $AFNIDIR/afni.sh
export FS_LICENSE="/tmp/freesurfer_license.txt"
BASE_WORK_DIR="/data/project/mb_decoder/work/fmriprep_work"
mkdir -p "$BASE_WORK_DIR"

source /data/project/tools/juseless_tools/fmriprep_24.1.1/fmriprep.sh

# Subjects running in parallel 
MAX_PARALLEL=3
# for subj in $(ls $bids_root_dir | grep "^sub"); do
# #for subj in ${SUBJECT}; do
run_subject () {
    subj="$1"
    subj_id="${subj#sub-}" 
    output_path="$output_dir/sub-${subj_id}"
    work_dir="${BASE_WORK_DIR}/sub-${subj_id}" 
    
    if [ -d "$output_path" ]; then
        echo "Skipping $subj - output already exists: $output_path"
        return
    fi
    
    mkdir -p "$work_dir"

    echo "Running fMRIPrep for subject: $subj (work_dir=$work_dir)"
    
    fmriprep $bids_root_dir $output_dir \
        participant \
        --participant-label "$subj_id" \
        --fs-license-file "$FS_LICENSE" \
        --output-spaces MNI152NLin2009cAsym:res-2 T1w \
        --work-dir "$work_dir" \
        --n_cpus 8 \
        --nprocs 1 \
        --omp-nthreads 8 \
        --random-seed 12345 \
        --skip-bids-validation \
        --skull-strip-fixed-seed \
        --stop-on-first-crash \
        --fs-no-reconall \
        --write-graph \
        --verbose \
        --notrack \
        --mem-mb 8000 \
        > "${BASE_WORK_DIR}/${subj}.log" 2>&1
    
    echo "Finished fMRIPrep for subject: $subj"
    echo "Cleaning up temporary files for $subj"
    rm -rf "$work_dir" 
}

export -f run_subject
export bids_root_dir output_dir FS_LICENSE BASE_WORK_DIR

for subj in $(ls $bids_root_dir | grep "^sub"); do
    ( run_subject "$subj" ) &

    # throttle: wait if we've hit MAX_PARALLEL running jobs
    while [ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n
    done
done

wait   
echo "Script completed for all subjects."
