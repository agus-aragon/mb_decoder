###
# INPUT:
#   1. bids_root_dir: Path to BIDS dataset  
#   2. output_dir: Derivative output directory 
#   3. FS_LICENSE: FreeSurfer license path 
#   4. WORK_DIR: Temporary working directory 
#
# DESCRIPTION:
#   Runs fMRIPrep (v24.1.1) subject-wise with:
#   - Slice timing ignored (TR=1.17s)
#   - MNI152NLin2009cAsym (2mm) + T1w output
#   - Deterministic skull-stripping
#   - No FreeSurfer recon (--fs-no-reconall)
#
# OUTPUT:
#   Preprocessed fMRI data in BIDS derivatives format:
#   - ${output_dir}/sub-XX/func/*preproc*.nii.gz
#   - Confounds TSV files
#   - Anatomical coregistration files
###
conda init
conda activate mb_decoder
SUBJECT="$1"

export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow
echo "Initializing script"
export FREESURFER_HOME="/data/project/tools/juseless_tools/freesurfer_7.4.1"
source $FREESURFER_HOME/freesurfer.sh
bids_root_dir=/data/project/mb_decoder/data/bids/mb_decoder/
output_dir=/data/project/mb_decoder/data/bids/mb_decoder/derivatives/
export FSLOUTPUTTYPE=NIFTI
cp ~/freesurfer_license.txt /tmp/freesurfer_license.txt
export FSLDIR="/data/project/tools/juseless_tools/fsl_6.0.4-patched2/"
source $FSLDIR/etc/fslconf/fsl.sh
export ANTSPATH="/data/project/tools/juseless_tools/ants_2.5.0/bin/"
export PATH=$ANTSPATH:$PATH
export PATH="/data/project/tools/juseless_tools/afni_24.3.06:$PATH"
export FS_LICENSE="/tmp/freesurfer_license.txt"
WORK_DIR="/data/project/mb_decoder/work/fmriprep_work"
mkdir -p "$WORK_DIR"

source /data/project/tools/juseless_tools/fmriprep_24.1.1/fmriprep.sh

# for subj in $(ls $bids_root_dir | grep "^sub"); do
for subj in ${SUBJECT}; do

    echo "Running fMRIPrep for subject: $subj"
    
    fmriprep $bids_root_dir $output_dir \
        participant \
        --participant-label ${subj#sub-} \
        --fs-license-file "$FS_LICENSE" \
        --output-spaces MNI152NLin2009cAsym:res-2 T1w \
        --work-dir "$WORK_DIR" \
        --n_cpus 8 \
        --nprocs 1 \
        --omp-nthreads 8 \
        --random-seed 12345 \
        --skip-bids-validation \
        --skull-strip-fixed-seed \
        --stop-on-first-crash \
        --fs-no-reconall \
        --ignore slicetiming \
        --write-graph \
        --verbose \
        --notrack \
        --mem-mb 8000
    
    echo "Cleaning up temporary files for $subj"
    rm -rf "$WORK_DIR"/*
    find /tmp -user $(whoami) -name "fmriprep*" -exec rm -rf {} \; 2>/dev/null
    
    echo "Finished fMRIPrep for subject: $subj"
done

echo "Script completed for all subjects."
