# %%
import pandas as pd
import numpy as np
from pathlib import Path
import json

data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder")
events_pattern = "**/func/*_task-ES_events.tsv"
out_path_events = data_path / "derivatives" / "events"
out_path_events.mkdir(parents=True, exist_ok=True)

TR = 1.5

# %% Extract events timepoints
all_events = []
all_events_df = pd.DataFrame()

for events_fname in data_path.glob(events_pattern):
    print(events_fname)
    subject = events_fname.name.split("_")[0]
    df_events = pd.read_csv(events_fname, sep="\t")
    df_events = df_events.sort_values("onset")

    this_events = {
        "subject": [],
        "timepoint": [],
        "event": [],
        "seconds_to_probe": [],
        "response_probe": [],
        "rt_prompt": [],
        "response_arousal": [],
        "rt_arousal": []
    }

    for idx, row in df_events.iterrows():
        probe_onset_in_TR = row["onset"] / TR
#TODO: end of trial needs arousal or end of task         

        this_events["subject"] += subject * XXX,
        this_events["timepoint"] += [],
        this_events["event"] += [],
        this_events["seconds_to_probe"] += [],
        this_events["response_probe"] += [row["response_mental_state"]] * XXX,
        this_events["rt_prompt"] += [row['response_time_mental_state']] * XXX,
        this_events["response_arousal"] += [row["response_arousal"]] * XXX,
        this_events["rt_arousal"] += [row['response_time_arousal']] * XXX,




