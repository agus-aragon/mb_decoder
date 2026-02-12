# %%
from pathlib import Path

import cleaner
import mne

# %%

db_path = Path("/Users/fraimondo/data/mb_decoder")
deriv_path = db_path / "derivatives" / "eeg_cleaner"

for epochs_fname in deriv_path.glob("**/*_eeg_epo.fif"):
    task = epochs_fname.name.split("_")[1].replace("task-", "")
    subject = epochs_fname.name.split("_")[0]
    print(f"Processing {subject} {task}")
    epochs = mne.read_epochs(epochs_fname)
    cleaner.reject(epochs_fname, epochs, required=True)

    ica_fname = epochs_fname.name.replace("_epo.fif", "_epo-ica.fif")
    ica_fname = epochs_fname.parent / ica_fname
    ica = mne.preprocessing.read_ica(ica_fname, verbose=True)
    cleaner.reject(epochs_fname, ica, required=True)

    ica.apply(epochs, exclude=ica.exclude, verbose=True)
    cleaned_fname = ica_fname.name.replace("_epo-ica.fif", "_ica-cleaned_epo.fif")
    cleaned_fname = ica_fname.parent / cleaned_fname
    print(f"Saving  to {ica_fname.parent / cleaned_fname}...")

    # Add average reference projection
    epochs.set_eeg_reference("average")

    # Interpolate bad channels
    epochs.interpolate_bads(reset_bads=True)

    epochs.save(ica_fname.parent / cleaned_fname, overwrite=True)

# %%
