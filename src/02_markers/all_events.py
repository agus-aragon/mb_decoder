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
        # Define probe onset
        thisevent_probe_onset_in_TR = probe_onsets_in_TR[idx]

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
        )

        total_timepoints = len(this_event_timepoints_TR)

        # Seconds to probe
        thisevent_seconds_to_probe = (
            this_event_timepoints_TR - thisevent_probe_onset_in_TR
        )
        thisevent_n_timepoints_before_probe = (thisevent_seconds_to_probe < 0).sum()

        thisevent_n_timepoints_after_probe = (thisevent_seconds_to_probe > 0).sum()

        # Save this event information
        this_events["subject"] += [subject] * total_timepoints

        this_events["timepoint"] += this_event_timepoints_TR.tolist()

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

        this_events["response_arousal"] += [row["response_arousal"]] * total_timepoints
        
        this_events["rt_arousal"] += [
            row["response_time_arousal"]
        ] * total_timepoints

    # Join subject data and all subject data
    subject_events_df = pd.DataFrame(this_events)
    all_events.append(subject_events_df)

all_events_df = pd.concat(all_events)
all_events_df.set_index(["subject", "timepoint"], inplace=True)
print(all_events_df.shape)
print(all_events_df.head())

# %%
