# %%
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mne
from mne.io import read_raw_eeglab
from matplotlib.lines import Line2D

project_path = Path("/data/project/mb_decoder/")
data_path = project_path /"data" / "bids"/ "mb_decoder"
eeglab_fmriartem_path = data_path / "derivatives" / "eeglab_fmriartrem"

out_path = project_path / "quality_check"
out_path.mkdir(parents=True, exist_ok=True)
# %%
all_impedances = []
for raw_file in data_path.glob("**/*_eeg.vhdr"):
    # Only read the metadata from task or rest, they are equal (eg., task)
    if "ES" not in raw_file.name:
        continue
    print(f"Reading {raw_file} ...   ")

    metadata = mne.io.read_raw_brainvision(vhdr_fname=raw_file, preload=False)
    impedances = pd.DataFrame(metadata.impedances).T
    impedances.reset_index(inplace=True)
    impedances.rename(columns={"index": "channel"}, inplace=True)

    impedances.drop(
        impedances[
            impedances["channel"].isin(
                ["65+", "65-", "66+", "66-", "67+", "67-", "68+", "68-"]
            )
        ].index,
        inplace=True,
    )  # drop CWL channels

    subj_impedances = impedances[["channel", "imp"]]
    subj_impedances.loc[:, "subject"] = raw_file.parent.parent.name
    all_impedances.append(subj_impedances)
impedances_df = pd.concat(all_impedances, ignore_index=True)


# %% Plot
# Boxplot, each channel a box, each subject a dot with different color
# Exclude subjext 3 as there was an issue when recording impedances
impedances_df = impedances_df[impedances_df["subject"] != "sub-003"]
plt.figure(figsize=(15, 6))
sns.boxplot(x="channel", y="imp", data=impedances_df, color="lightgrey", showfliers=False) 
sns.stripplot(x="channel", y="imp", data=impedances_df, hue="subject", dodge=True, jitter=True, size=5)

plt.xticks(rotation=90)
plt.ylim(-1,50)
plt.xlabel("Channel", fontsize=16)
plt.ylabel("Impedance (kOhm)",fontsize=16)
plt.title("Impedance Distribution Across Channels and Subjects", fontsize=18)
plt.text(
    1,
    47,
    "One subject not ploted because of bug in recording impedances (all the same)",
    fontsize=16
)
plt.legend().set_visible(False)
plt.tight_layout()
# plt.show()
plt.savefig(out_path / "impedances.png")
plt.close()

# %% Median of impedances per electrode
channel_medians = impedances_df.groupby("channel")["imp"].median()
channel_means = impedances_df.groupby("channel")["imp"].mean()

# %% Cardiobalistic information in CWL
channel_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] 
all_results = {} 


for eeg_file in eeglab_fmriartem_path.glob("**/*_task-ES_desc-fmriClean_eeg.set"):
    subject = eeg_file.stem.split("_")[0]
    print(f"Processing {subject}...")
    raw = read_raw_eeglab(eeg_file, preload=True)
    cwl_names = raw.ch_names[64:68]

    raw_cwl = raw.copy().pick(cwl_names)
    psd = raw_cwl.compute_psd(fmin=0.1, fmax=10, method="welch", n_fft=2048, verbose=False)
    psd_data, freqs = psd.get_data(return_freqs=True)  # shape (4, n_freqs)
    psd_db = 10 * np.log10(psd_data * 1e12)  # convert to dB re 1 uV^2, matches MNE's default plot scaling
 
    all_results[subject] = {
        "freqs": freqs,
        "psd_db": psd_db,
        "ch_names": cwl_names,
    }

fig, ax = plt.subplots(figsize=(14, 7))
 
for subject, res in all_results.items():
    freqs = res["freqs"]
    psd_db = res["psd_db"]
    for ch_idx in range(psd_db.shape[0]):
        ax.plot(
            freqs,
            psd_db[ch_idx],
            color=channel_colors[ch_idx % len(channel_colors)],
            alpha=0.3,
            linewidth=2,
        )

legend_labels = res["ch_names"] if all_results else [f"CWL{i+1}" for i in range(4)]
legend_elements = [
    Line2D([0], [0], color=channel_colors[i], lw=2, label=legend_labels[i])
    for i in range(len(channel_colors))
]
ax.legend(handles=legend_elements, loc="upper right")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB/Hz re 1 µV²)")
ax.set_title(f"CWL channel PSDs — all subjects (n={len(all_results)})")
ax.grid(True, linestyle=":", alpha=0.5)
 
plt.tight_layout()
plt.savefig(out_path / "CWL_PSD_all_subjects.png")
plt.show()
# %%
