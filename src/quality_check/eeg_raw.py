# %%
import mne
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder")

# %%
all_impedances = []
for raw_file in db_path.glob("**/*_eeg.vhdr"):
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
sns.stripplot(x="channel", y="imp", data=impedances_df, hue="subject", dodge=True, jitter=True, size=7)
plt.xticks(rotation=90)
plt.ylim(-1,50)
plt.xlabel("Channel", fontsize=16)
plt.ylabel("Impedance (kOhm)",fontsize=16)
plt.title("Impedance Distribution Across Channels and Subjects", fontsize=18)
plt.legend().set_visible(False)
plt.tight_layout()
plt.show()

# %%
