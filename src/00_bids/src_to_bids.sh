#!/bin/bash

SUBJECT="001"

heudiconv \
  -d /data/project/mb_decoder/data/subj_raw/sub-{subject}/fMRI/*_MR/*.dcm \
  -o /data/project/mb_decoder/data/bids/mb_decoder/ \
  -f /home/agusaragon/dev/mb_decoder/src/00_bids/heuristic.py \
  -s ${SUBJECT} \
  -c dcm2niix \
  -b \
  --overwrite

datalad save -m "Converted subject ${SUBJECT} to BIDS format"