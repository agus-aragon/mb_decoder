# %%
from pathlib import Path

import cleaner
import mne
import numpy as np
from common import _bp_filter


# %%

db_path = Path("/Users/fraimondo/data/mb_decoder")
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

    # TODO: Cut data
    if task == "rest":
        # drop 10 triggers with value 1
        # drop data after last trigger with value 1 + 1.5s (TR)
        events = mne.find_events(raw)

        mri_volumes_events = events[events[:, 2] == 1]

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

        raw.crop(start / raw.info["sfreq"], end / raw.info["sfreq"])

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
    else:
        raise NotImplementedError(f"Task {task} not implemented yet")

    # Save the epochs
    out_fname = raw_file.with_stem(f"{raw_file.stem}_epo")
    epochs.save(out_fname, overwrite=True)

# %%
