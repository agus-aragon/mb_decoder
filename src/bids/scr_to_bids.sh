#!/bin/bash

SUBJECT="001"


heudiconv \
  -d /data/project/mb_decoder/data/pilot/sub-{subject}/fMRI/dicom/*_MR/*.dcm \
  -o /data/project/mb_decoder/data/pilot/sub-001/fMRI/bids \
  -f /home/agusaragon/dev/mb_decoder/src/bids/heuristic.py \
  -s ${SUBJECT} \
  -c dcm2niix \
  -b \
  --overwrite