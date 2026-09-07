# %%
from pathlib import Path
import mne
import pandas as pd

db_path = Path(
    "/data/project/mb_decoder/data/bids/mb_decoder/derivatives/eeglab_fmriartrem"
)
deriv_path = db_path.parent / "eeg_cleaner"
deriv_path.mkdir(parents=True, exist_ok=True)

# %%
for raw_file in db_path.glob("**/*.set"):
    out_fname = deriv_path / raw_file.relative_to(db_path).with_suffix(".fif")
    if out_fname.exists():
        print(
            f"Subject {raw_file.parent.parent.name} file already exists - Skipping  // {out_fname}. "
        )
        continue
    out_fname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading {raw_file} ...   ")
    raw = mne.io.read_raw_eeglab(raw_file, preload=True)

    raw.drop_channels(["65", "66", "67", "68"])  # Get rid of CWL

    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(montage)

    # Modify events for task-ES
    if "task-ES" in raw_file.name:
        psychopy_path = (
            raw_file.parent.parent.parent.parent.parent
            / raw_file.parent.parent.name
            / "func"
            / f"{raw_file.parent.parent.name}_task-ES_events.tsv"
        )
        psychopy = pd.read_csv(psychopy_path, sep="\t")
        events, event_id = mne.events_from_annotations(raw)
        sfreq = raw.info["sfreq"]
        seconds = events[:, 0] / sfreq

        # Create DataFrame
        df = pd.DataFrame(
            {
                "sample": events[:, 0],
                "seconds": seconds,
                "previous_sample": events[:, 1],  # Usually 0 for annotations
                "event_id": events[:, 2],
            }
        )

        # Map numeric event codes to descriptions 
        df["description"] = df["event_id"].map({v: k for k, v in event_id.items()})

        new_descriptions = df["description"].copy()

        for trig_desc in ["response_probe", "response_arousal", "start_trial", "probe"]:
            mask = df["description"] == trig_desc
            n_matches = mask.sum()

            if n_matches != len(psychopy):
                print(
                    f"  WARNING: {n_matches} EEG '{trig_desc}' events vs "
                    f"{len(psychopy)} psychopy trials"
                )
                continue

            idx = df.index[mask]
            for i, row_idx in enumerate(idx):
                trial_num = i + 1
                mental_state = psychopy.loc[i, "response_mental_state"]
                if trig_desc == "response_probe":
                    new_descriptions.loc[row_idx] = (
                        f"Response/Probe/Trial{trial_num}/{mental_state}"
                    )
                elif trig_desc == "start_trial":
                    new_descriptions.loc[row_idx] = (
                        f"Start/Trial{trial_num}"
                    )
                elif trig_desc == "probe":
                    new_descriptions.loc[row_idx] = (
                        f"Probe/Trial{trial_num}"
                    )
                else:  # response_arousal
                    arousal_val = psychopy.loc[i, "response_arousal"]
                    new_descriptions.loc[row_idx] = (
                        f"Response/Arousal/Trial{trial_num}/{mental_state}/{arousal_val}"
                    )

        df["description"] = new_descriptions

        raw.annotations.description = new_descriptions.to_numpy().astype(str)

        # Sanity check: confirm the new event_id mapping looks right
        _, new_event_id = mne.events_from_annotations(raw)
        print(f"  New event types: {(new_event_id.keys())}")
    print(f"Saving to FIF format in {out_fname}...   ")
    raw.save(out_fname, overwrite=True)

