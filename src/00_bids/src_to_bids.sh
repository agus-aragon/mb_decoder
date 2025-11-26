########################################################
## Start by organizing raw data like:                 ##  
## /data/project/mb_decoder/data/subj_raw/sub-XXX/    ##
## fMRI/ -> DICOMs + Physio                           ## 
## eeg/ -> BrainVision files                          ##
## psychopy/ -> behavioral data files                 ##
########################################################

#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mb_decoder
SUBJECT="$1"  # e.g., 001 (console usage: ./src_to_bids.sh 001)
SUBJECT_ID_SCANNER = "$2" # e.g., 4472 (from fmriserver)

BASE_DIR="/data/project/mb_decoder"
RAW_DIR="${BASE_DIR}/data/subj_raw"
BIDS_DIR="${BASE_DIR}/data/bids/mb_decoder"
HOME_DIR="/home/agusaragon"
echo "Converting sub-${SUBJECT} to BIDS format..."

######################## fMRI ########################
# Convert DICOMs to BIDS using Heudiconv
echo "Working on fMRI..."
heudiconv \
  -d "${RAW_DIR}/sub-{subject}/fMRI/*_MR/*.dcm" \
  -o "${BIDS_DIR}" \
  -f "${HOME_DIR}/dev/mb_decoder/src/00_bids/heuristic.py" \
  -s ${SUBJECT} \
  -c dcm2niix \
  -b \
  --overwrite
echo "fMRI conversion complete."

######################## Pshyio ########################
#  Convert Physio DICOMs to BIDS using dcm2bidsphysio
echo "Working on Physio..."
dcm2bidsphysio --infile "${RAW_DIR}/sub-${SUBJECT}/fMRI/Agustina_StdPE-70min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_PhysioLog_16_MR/1.dcm" \
               --bidsprefix "${BIDS_DIR}/sub-${SUBJECT}/func/sub-${SUBJECT}_task-ES"
dcm2bidsphysio --infile "${RAW_DIR}/sub-${SUBJECT}/fMRI/Agustina_StdPE-10min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_PhysioLog_12_MR/1.dcm" \
               --bidsprefix "${BIDS_DIR}/sub-${SUBJECT}/func/sub-${SUBJECT}_task-rest"
echo "Physio conversion complete."

######################## EEG ########################
echo "Working on EEG..."

# Modify Events in .vmrk files
cd "${RAW_DIR}/sub-${SUBJECT}/eeg/"
  # 1. Treat scanner toggle on/off as a single event "Scanner"
sed -i 's/T  1_on/Scanner/g; s/T  1_off/Scanner/g' sub-${SUBJECT}_task-ES_eeg.vmrk
sed -i 's/T  1_on/Scanner/g; s/T  1_off/Scanner/g' sub-${SUBJECT}_task-rest_eeg.vmrk
  # 2. Task ES events
sed -i 's/S 16/start_trial/g;
  s/Stimulus,S  8/Response,response_arousal/g;
  s/Stimulus,S  4/Response,response_probe/g;
  s/S  2/probe/g;
  s/Stimulus,S  1/EEG,start_eeg/g;
  s/Stimulus,S 32/EEG,end_eeg/g
' sub-${SUBJECT}_task-ES_eeg.vmrk

# Convert EEG to BIDS using BVTools (Download Dotnet + clone repo first)
cd "${HOME_DIR}/dev/BVTools/"
export DOTNET_ROOT=$HOME/.dotnet
export PATH=$PATH:$HOME/.dotnet
dotnet run --project src/FileFormats/src/FileFormats.BrainVisionToBidsConverterCLI/FileFormats.BrainVisionToBidsConverterCLI.csproj -- \
  --bv-header-file "${RAW_DIR}/sub-${SUBJECT}/eeg/sub-${SUBJECT}_task-ES_eeg.vhdr" \
  --bids-destination-folder "${BIDS_DIR}" \
  --subject ${SUBJECT} \
  --task "ES"
dotnet run --project src/FileFormats/src/FileFormats.BrainVisionToBidsConverterCLI/FileFormats.BrainVisionToBidsConverterCLI.csproj -- \
  --bv-header-file "${RAW_DIR}/sub-${SUBJECT}/eeg/sub-${SUBJECT}_task-rest_eeg.vhdr" \
  --bids-destination-folder "${BIDS_DIR}" \
  --subject ${SUBJECT} \
  --task "rest"
echo "EEG conversion complete."

######################### Psychopy events ########################
python "${HOME_DIR}/dev/mb_decoder/src/00_bids/psychopy_to_bids.py" ${SUBJECT} ${RAW_DIR} ${BIDS_DIR}


######################### Add to sourcedata ########################
echo "s0${SUBJECT_ID_SCANNER}, sub-${SUBJECT}" >> "${BASE_DIR}/data/sourcedata/mapping.csv"


######################### Save to datalad ########################
# Save changes to datalad
cd "${BIDS_DIR}"
echo Saving changes to datalad...
datalad save -m "Converted subject ${SUBJECT} to BIDS format"
echo "Conversion to BIDS complete for sub-${SUBJECT}"