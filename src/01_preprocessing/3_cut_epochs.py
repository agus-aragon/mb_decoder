# %%
from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent
external_path = current_dir.parent / "external" / "eeg_cleaner"
if str(external_path) not in sys.path:
    sys.path.append(str(external_path))
import cleaner
import re
import mne
import numpy as np
import pandas as pd
from common import _bp_filter


# %%

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder")
deriv_path = db_path / "derivatives" / "eeg_cleaner"

volumes_to_drop_start = 10
epoch_length = 2  # cut into two seconds epochs
filter_params = {"lpass": 45.0, "hpass": 0.5}


# %%
for raw_file in deriv_path.glob("**/*_eeg.fif"):
    task = raw_file.name.split("_")[1].replace("task-", "")
    subject = raw_file.name.split("_")[0]

    print(f"Processing {subject} {task}")
    raw = mne.io.read_raw_fif(raw_file, preload=True)
    cleaner.reject(raw_file, raw, required=True)

    # Filter the data
    _bp_filter(raw, params=filter_params, n_jobs=-1)

    # Cut data
    # drop 10 triggers with value 1
    # drop data after last trigger with value 1 + 1.5s (TR)
    events, event_id = mne.events_from_annotations(raw)

    mri_volumes_events = events[events[:, 2] == event_id['Scanner']]

    # Detect the TR
    tr_samps = np.unique(np.diff(mri_volumes_events[:, 0]))

    if len(tr_samps) > 1:
        raise ValueError(
            f"Multiple TRs detected: {tr_samps / raw.info['sfreq']} s"
        )
    tr_samps = tr_samps[0]
    tr = tr_samps / raw.info["sfreq"]
    print(f"Detected TR: {tr:.2f} s ({tr_samps} samples)")

    # Drop the first volumes, we start at the next
    start = mri_volumes_events[volumes_to_drop_start, 0]

    # We finish after the last volume + TR
    end = mri_volumes_events[-1, 0] + tr_samps - 1
    # Some recordings stop during the last volume: end with the EEG instead
    if end > raw.last_samp:
        print(
            f"EEG ends {(end - raw.last_samp) / raw.info['sfreq']:.2f} s "
            "before the last volume + TR, cropping at the end of the EEG"
        )
        end = raw.last_samp

    raw.crop(
        (start - raw.first_samp) / raw.info["sfreq"],
        (end - raw.first_samp) / raw.info["sfreq"],
    )

    if task == "rest":
        # Cut into epochs
        new_events = mne.make_fixed_length_events(
            raw, id=1, duration=epoch_length
        )
        epochs = mne.Epochs(
            raw,
            new_events,
            tmin=0,
            tmax=epoch_length,
            baseline=None,
            preload=True,
        )
    elif task == "ES":
        # Cut each trial (trial start -> probe) into epochs aligned backwards
        # from the probe, so no epoch crosses a probe.
        # Epochs are only for cleaning; analysis windows are chosen later
        # with the metadata (trial, time_to_probe).
        events, event_id = mne.events_from_annotations(raw)
        id_to_desc = {v: k for k, v in event_id.items()}
        descs = np.array([id_to_desc[x] for x in events[:, 2]])
        is_start = np.array([d.startswith("start") for d in descs])
        is_probe = np.array(
            [d == "probe" or d.startswith("probe/") for d in descs]
        )
        start_samps = events[is_start, 0]
        probe_samps = events[is_probe, 0]
        probe_descs = descs[is_probe]

        if len(probe_samps) == 0:
            print(f"WARNING: no probe events for {subject} {task}, skipping")
            continue

        epoch_samps = int(round(epoch_length * raw.info["sfreq"]))
        onsets, rows = [], []
        for i_probe, (probe_samp, desc) in enumerate(
            zip(probe_samps, probe_descs)
        ):
            match = re.search(r"trial(\d+)$", desc)
            trial = int(match.group(1)) if match else i_probe + 1
            # Keep only epochs within the trial and the cropped data.
            # The first trial starts before the crop, so its start marker
            # may be gone: use the beginning of the cropped data.
            prev_starts = start_samps[start_samps < probe_samp]
            trial_start = (
                prev_starts[-1] if len(prev_starts) else raw.first_samp
            )
            trial_start = max(trial_start, raw.first_samp)
            onset = probe_samp - epoch_samps
            while onset >= trial_start:
                onsets.append(onset)
                rows.append(
                    {
                        "trial": trial,
                        "time_to_probe": (onset - probe_samp)
                        / raw.info["sfreq"],
                        "overlap": False,
                    }
                )
                onset -= epoch_samps
            # The leftover (< 2 s) at the start of the trial would be lost:
            # cover it with one epoch starting at the trial start, which
            # overlaps with the next one.
            leftover = onset + epoch_samps - trial_start
            if leftover > 0 and probe_samp - trial_start >= epoch_samps:
                onsets.append(trial_start)
                rows.append(
                    {
                        "trial": trial,
                        "time_to_probe": (trial_start - probe_samp)
                        / raw.info["sfreq"],
                        "overlap": True,
                    }
                )

        order = np.argsort(onsets)
        new_events = np.column_stack(
            [
                np.array(onsets)[order],
                np.zeros(len(onsets), int),
                np.ones(len(onsets), int),
            ]
        )
        metadata = pd.DataFrame(rows).iloc[order].reset_index(drop=True)
        epochs = mne.Epochs(
            raw,
            new_events,
            tmin=0,
            tmax=epoch_length,
            baseline=None,
            metadata=metadata,
            preload=True,
        )
        print(
            f"{len(epochs)} epochs from {metadata['trial'].nunique()} trials"
        )
    else:
        raise NotImplementedError(f"Task {task} not implemented yet")


    # Save the epochs
    out_fname = raw_file.with_stem(f"{raw_file.stem}_epo")
    epochs.save(out_fname, overwrite=True)

# %%
