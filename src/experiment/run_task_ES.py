###
# Run Task "Experience Sampling"
# INPUT:
#
#
#
# DESCRIPTION:
# This script runs the Mind Blanking Experience Sampling task in the scanner.
#
# OUTPUT:
# - A log file with the task results.

# Conda environment: psychopy_env
###

#!/usr/bin/env python3

#%% Initialization

import logging 
import random
import os
import csv
from psychopy import visual, core, event, sound, parallel
from task_ES import experience_sampling
# %%
if __name__ == "__main__":
    params = {
        'subj': '001',
        'n_trials': 3,
        'interval': 5,  # seconds between trials
        'jittering':2,
        'states': ['Thought', 'Mind Blanking', 'Asleep'],
        'parallel': False,
        'response_buttons': ['1', '2', '3']
    }
    
    experiment = experience_sampling(params)
    experiment.run_experiment()