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

# Iterate through subjects
for events_fname in data_path.glob(events_pattern):
    # Load subject events.tsv
    print(events_fname)
    subject = events_fname.name.split("_")[0]
    df_events = pd.read_csv(events_fname, sep="\t")
    df_events = df_events.sort_values("onset")

    this_events = {
        "subject": [],
        "timepoint": [],
        "n_trial": [],
        "event": [],
        "seconds_to_probe": [],
        "response_prompt": [],
        "rt_prompt": [],
        "response_arousal": [],
        "rt_arousal": [],
    }

    # Extract all probe onset and start of trial
    probe_onsets_in_TR = (df_events["onset"] / TR).astype("int")
    first_timepoints_trial_TR = (
        (df_events["onset"] - df_events["rest_duration"]) / TR
    ).astype("int")

    # Iterate though each event
    for idx, row in df_events.iterrows():
        # Trial number
        thisevent_trial_number = idx + 1  # to acccount for python 0-indexing

        # Define probe onset
        thisevent_probe_onset_in_seconds = df_events["onset"][idx]

        # Define start of trial
        thisevent_first_timepoints_trial_TR = first_timepoints_trial_TR[idx]

        # Define end of trial (= start of next trial)
        if idx == df_events.index[-1]:
            lastevent_extra_time_responses = (
                df_events.loc[idx, "response_time_mental_state"]
                + df_events.loc[idx, "response_time_arousal"]
                + 5
            )  # 5 seconds to account for lag between probe and prompt, and arousal prompt + rating (= 3 TRs)

            thisevent_end_timepoint_trial_TR = (
                df_events.loc[idx, "onset"] + lastevent_extra_time_responses
            ) / TR
        else:
            thisevent_end_timepoint_trial_TR = first_timepoints_trial_TR[
                idx + 1
            ]

        # Define timepoints before and after probe separatedly for
        this_event_timepoints_TR = np.arange(
            thisevent_first_timepoints_trial_TR,
            thisevent_end_timepoint_trial_TR,
            1,
        ).astype("int")

        total_timepoints = len(this_event_timepoints_TR)

        # Seconds to probe
        thisevent_seconds_to_probe = (
            this_event_timepoints_TR * TR - thisevent_probe_onset_in_seconds
        )
        thisevent_n_timepoints_before_probe = (
            thisevent_seconds_to_probe <= 0
        ).sum() - 1

        thisevent_n_timepoints_after_probe = (
            thisevent_seconds_to_probe > 0
        ).sum()

        # Save this event information
        this_events["subject"] += [subject] * total_timepoints

        this_events["timepoint"] += this_event_timepoints_TR.tolist()

        this_events["n_trial"] += [thisevent_trial_number] * total_timepoints

        this_events["event"] += (
            ["rest"] * thisevent_n_timepoints_before_probe
            + ["probe"]
            + ["response"] * thisevent_n_timepoints_after_probe
        )
        this_events["seconds_to_probe"] += thisevent_seconds_to_probe.tolist()

        this_events["response_prompt"] += [
            row["response_mental_state"]
        ] * total_timepoints

        this_events["rt_prompt"] += [
            row["response_time_mental_state"]
        ] * total_timepoints

        this_events["response_arousal"] += [
            row["response_arousal"]
        ] * total_timepoints

        this_events["rt_arousal"] += [
            row["response_time_arousal"]
        ] * total_timepoints

    # Join subject data and all subject data
    subject_events_df = pd.DataFrame(this_events)
    all_events.append(subject_events_df)

all_events_df = pd.concat(all_events)

## Checks

# Are there repeated timepoints per subject? (There should not)
all_events_df.duplicated(subset=["subject", "timepoint"]).value_counts()

# Do all trials only have one (1) probe? (There should be only 1 probe per trial)
n_probes_per_trial = 1
(
    (
        all_events_df[all_events_df["event"] == "probe"]
        .groupby(["subject", "n_trial"])
        .size()
    )
    > n_probes_per_trial
).sum()

# Is there any gap between timepoints? (There should not be)
all_events_df.groupby('subject')['timepoint'].diff().value_counts()

# Does it contain all subjects? (n=50)
len(all_events_df['subject'].unique())

# Any missing value? (There should not be)
all_events_df.isna().sum()

# Export
all_events_df.set_index(["subject", "timepoint"], inplace=True)
all_events_df.to_csv(out_path_events / "all_events.csv")

# %% Side .json with metadata/explanation

events_json = {
    "subject": {
        "LongName": "Subject ID",
        "Description": "unique identifier for each participant",
    },
    "timepoint": {
        "LongName": "",
        "Description": "count of MRI volumes (already syncronized)",
    },
    "n_trial": {
        "LongName": "Number of Trial",
        "Description": "trial ID/number, per subject. Each subject has 50 trials",
    },
    "event": {
        "LongName": "Type of event",
        "Description": "explains what was happening at those timepoints",
        "Levels": {
            "rest": "participant was looking at the fixation cross, letting their mind free (resting state)",
            "probe": "participant was probed to report their inmmediate mental state ('!' visual stimuli + sound)",
            "response": "participant was replying the mental state and the arousal prompt",
        },
    },
    "seconds_to_probe": {
        "LongName": "Seconds before or after probe",
        "Description": "Fixating probe at time 0, each timepoint gets assigned a negative (seconds before probe) or positive (seconds after probe) value. First timepoint is the start of the trial, last timepoint is the start of the next trial",
    },
    "response_prompt": {
        "LongName": "Response Mental state prompt",
        "Description": "Participant response to the mental state prompt (4 options)",
        "Levels": {
            "Thought": "Thinking about something",
            "Blank": "Mind was blank, no though you can spot",
            "Sleep": "Feeling drowsy or asleep",
            "Sensations": "Noticing the environment or body sensations",
        },
    },
    "rt_prompt": {
        "LongName": "Reaction time to mental state prompt",
        "Description": "Time the participant took to choose their mental state since they were presented with the options to report (prompt)",
    },
    "response_arousal": {
        "LongName": "Response to arousal question",
        "Description": "Participant report of their arousal levels from 0% (very sleepy) to 100% very alert",
    },
    "rt_arousal": {
        "LongName": "Reaction time to arousal question",
        "Description": "Time the participant took to choose their arousal level since they were presented with the scale to report",
    },
}

name_json = "all_events.json"

with open(out_path_events / name_json, "w") as f:
    json.dump(events_json, f, indent=4)
