# %% Initialization
import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path

subj = sys.argv[1]# '001' 
raw_dir = Path(sys.argv[2])#Path("/data/project/mb_decoder/data/dicom/pilots")
bids_dir = Path(sys.argv[3])#Path("/data/project/mb_decoder/data/bids/pilots")
datapath = raw_dir / f"sub-{subj}" / "psychopy"

# Load all events file 
with open(datapath / "task-ES" / f"sub_{subj}_task-ES_ev.yaml", 'r') as file:
    events = pd.DataFrame(yaml.safe_load(file))

# Load task file 
with open(datapath / "task-ES" / f"sub_{subj}_task-ES_exp.yaml", 'r') as file:
    task = pd.DataFrame(yaml.safe_load(file))
task = task.set_index('trial_num')

# %% Time baseline correction (sync with fMRI)
first_trigger_onset = float(events[(events[1] == 'SCANNER') & (events[2] == 0)][0].iloc[0])

task['probe_onset'] = task['probe_onset'] - first_trigger_onset
task['prompt_onset'] = task['prompt_onset'] - first_trigger_onset

#%% BIDS format
events_tsv = pd.DataFrame()
events_tsv['onset'] = task['probe_onset']
events_tsv['duration'] = 0.0
events_tsv['trial_type'] = 'probe'
events_tsv['response_mental_state'] = task['state']
events_tsv['response_time_mental_state'] = task['response_rt']
events_tsv['response_arousal'] = task['arousal']
events_tsv['response_time_arousal'] = task['arousal_rt']
events_tsv['rest_duration'] = task['rest_duration']

name_tsv = f"sub-{subj}_task-ES_events.tsv"

#%% Create .json with metadata
events_json = {
    "TaskName": "ES",
    "TaskDescription": "Experience sampling (ES). Participants were probed (50 times, every 45 s with a ±15 s jitter) to report about their immediate mental content (Thought, Blank, Sleep, Sensations). After each trial, they rated their current arousal levels",
    "onset": {
        "Description": "Time of probe onset in seconds from first scanner trigger (baseline corrected)",
        "Units": "s"
    },
    "duration": {
        "Description": "Duration of the probe (beep) presentation.",
        "Units": "s"
    },
    "trial_type": {
        "Description": "Type of event. Here always 'probe' (visual and auditory stimulus (Exclamation mark “!” 1000 ms + Tone of 1000 Hz at 70 dB for 2 seconds (Van Calster et al., 2017))."
    },
    "response_mental_state": {
        "Description": "Reported mental state at probe (Thought/Blank/Sleep/Sensation)."
    },
    "response_time_mental_state": {
        "Description": "Reaction time for mental state response, from mental-state prompt onset (prompt ocurring miliseconds after probe).",
        "Units": "s"
    },
    "response_arousal": {
        "Description": "Arousal rating on a 0–100 scale (from 0=very sleepy to 100=very alert)."
    },
    "response_time_arousal": {
        "Description": "Reaction time for arousal rating, from arousal prompt onset.",
        "Units": "s"
    },
    "rest_duration": {
        "Description": "Rest interval (fixation cross) preceding this probe (interval + jitter).",
        "Units": "s"
    }
}
name_json = f"sub-{subj}_task-ES_events.json"

#%% Export
events_path = bids_dir / f"sub-{subj}" / "func" 
events_path.mkdir(parents=True, exist_ok=True)

with open(events_path / name_json, "w") as f:
    json.dump(events_json, f, indent=4)

events_tsv.to_csv(events_path / name_tsv, sep="\t", index=False)
