# %%
from pathlib import Path
import sys
current_dir = Path(__file__).resolve().parent
external_path = current_dir.parent / "external" / "eeg_cleaner"
if str(external_path) not in sys.path:
    sys.path.append(str(external_path))
import cleaner
import mne

# %%

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder")
deriv_path = db_path / "derivatives" / "eeg_cleaner"


for epochs_fname in deriv_path.glob("**/*_eeg_epo.fif"):
    task = epochs_fname.name.split("_")[1].replace("task-", "")
    subject = epochs_fname.name.split("_")[0]
    print(f"Processing {subject} {task}")
    epochs = mne.read_epochs(epochs_fname)
    cleaner.reject(epochs_fname, epochs, required=True)

    picks = mne.pick_types(
        epochs.info, meg=False, eeg=True, eog=False, stim=False, exclude="bads"
    )
    ica = mne.preprocessing.ICA(
        n_components=0.9999,
        max_iter="auto",
        method="picard",
        fit_params=dict(ortho=False, extended=True),
        verbose=True,
    )
    ica.fit(epochs, picks=picks, verbose=True)
    ica_fname = epochs_fname.name.replace("_epo.fif", "_epo-ica.fif")
    ica.save(epochs_fname.parent / ica_fname, overwrite=True)

# %%
