# %%
from pathlib import Path

import cleaner
import mne
from common import _bp_filter

# %%

db_path = Path("/Users/fraimondo/data/mb_decoder")
deriv_path = db_path / "derivatives" / "eeg_cleaner"

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
        pass

    # Save the epochs
    out_fname = raw_file.with_stem(f"{raw_file.stem}_epo")
    epochs.save(out_fname, overwrite=True)

# %%
