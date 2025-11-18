# %% Initialization
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

datapath =Path("/Users/agusaragon/Documents/psychopy") #  Path("/data/project/mb_decoder/data/subj_raw/")
subj = '001' #sys.argv[1]
datapath_subj = datapath #/ f"sub-{subj}" #/ "psychopy"

# Load all events file 
with open(datapath_subj / "task-ES" / f"sub_{subj}_task-ES_ev.yaml", 'r') as file:
    events = pd.DataFrame(yaml.safe_load(file))

# Load task file 
with open(datapath_subj / "task-ES" / f"sub_{subj}_task-ES_exp.yaml", 'r') as file:
    task = pd.DataFrame(yaml.safe_load(file))
task = task.set_index('trial_num')

# %% Time baseline correction (sync with fMRI)
first_trigger_onset = float( events[(events[1] == 'SCANNER') & (events[2] == 0)][0].iloc[0])

task['probe_onset'] = task['probe_onset'] - first_trigger_onset
task['prompt_onset'] = task['prompt_onset'] - first_trigger_onset

# probe_onset - (prompt_onset + arousal_rt + response_rt + rest_duration[+1]) = 2 seconds of difference! always
# diferencia entre probe y probe deberia ser = rest_duration + response_rt + arousal_rt

# %%
# BIDS format