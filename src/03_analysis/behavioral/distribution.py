# %%
import pandas as pd
from pathlib import Path

main_path = Path("/data/project/mb_decoder/")
db_path = main_path / "data" / "bids" / "mb_decoder"
out_path = main_path / "output" / "03_analysis" / "behavioral" / "distribution"
out_path.mkdir(parents=True, exist_ok=True)

states = ["Blank", "Sleep", "Thought", "Sensation"]

# %% Read all events.tsv files and concatenate them into a single DataFrame
all_events = []
for events_file in sorted(db_path.glob("**/func/*_task-ES_events.tsv")):
    event_subj_df = pd.read_csv(events_file, sep="\t")
    event_subj_df["subject"] = events_file.parent.parent.name
    all_events.append(event_subj_df)
events_df = pd.concat(all_events, ignore_index=True)
probes_df = events_df[events_df["trial_type"] == "probe"]

# %% Absolute and relative frequency of each mental state per subject
abs_freq = (
    pd.crosstab(probes_df["subject"], probes_df["response_mental_state"])
    .reindex(columns=states, fill_value=0)
)
# Relative to the number of answered probes of each subject
rel_freq = abs_freq.div(abs_freq.sum(axis=1), axis=0)

freq_df = (
    abs_freq.stack()
    .rename("abs_freq")
    .to_frame()
    .join(rel_freq.stack().rename("rel_freq"))
    .reset_index()
)
freq_df.to_csv(
    out_path / "mental_state_freq_per_subject.tsv", sep="\t", index=False
)
freq_df

# %% Wide format (one row per subject)
freq_wide = pd.concat({"abs": abs_freq, "rel": rel_freq}, axis=1)
freq_wide
# %%
