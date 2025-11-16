#!/bin/bash
conda init
conda activate mb_decoder
SUBJECT="$1"

echo "Converting sub-${SUBJECT} to BIDS format..."

# Convert DICOMs to BIDS 
heudiconv \
  -d /data/project/mb_decoder/data/subj_raw/sub-{subject}/fMRI/*_MR/*.dcm \
  -o /data/project/mb_decoder/data/bids/mb_decoder/ \
  -f /home/agusaragon/dev/mb_decoder/src/00_bids/heuristic.py \
  -s ${SUBJECT} \
  -c dcm2niix \
  -b \
  --overwrite

# Extract physio data
dcm2bidsphysio --infile /data/project/mb_decoder/data/subj_raw/sub-${SUBJECT}/fMRI/Agustina_StdPE-70min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_PhysioLog_16_MR/1.dcm \
               --bidsprefix /data/project/mb_decoder/data/bids/mb_decoder/sub-${SUBJECT}/func/sub-${SUBJECT}_task-ES

dcm2bidsphysio --infile /data/project/mb_decoder/data/subj_raw/sub-${SUBJECT}/fMRI/Agustina_StdPE-10min_cmrr_mb2ep2d_1GE_TR1p5s_3p0mm_PhysioLog_12_MR/1.dcm \
               --bidsprefix /data/project/mb_decoder/data/bids/mb_decoder/sub-${SUBJECT}/func/sub-${SUBJECT}_task-rest

datalad save -m "Converted subject ${SUBJECT} to BIDS format"
echo "Conversion complete for subject ${SUBJECT}"
