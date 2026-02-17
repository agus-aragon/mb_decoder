# %%
from pathlib import Path
import mne
# %%

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives/eeglab_fmriartrem")
deriv_path = db_path.parent / "eeg_cleaner"
deriv_path.mkdir(parents=True, exist_ok=True)

# %%
for raw_file in db_path.glob("**/*.set"):
    out_fname = deriv_path / raw_file.relative_to(db_path).with_suffix(".fif")
    out_fname.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading {raw_file} ...   ")
    raw = mne.io.read_raw_eeglab(raw_file, preload=True)

    raw.drop_channels(['65', '66', '67', '68']) # Get rid of CWL

    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(montage)
    print(f"Saving to FIF format in {out_fname}...   ")
    raw.save(out_fname, overwrite=True)

# %%
