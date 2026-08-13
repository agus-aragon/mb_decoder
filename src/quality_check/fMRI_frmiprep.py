# %%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


project_path = Path("/data/project/mb_decoder/")
fmriprep_path = (
    project_path / "data" / "bids" / "mb_decoder" / "derivatives" / "fmriprep"
)
out_path = project_path / "quality_check"
out_path.mkdir(parents=True, exist_ok=True)

task = "ES"  # 'rest'
# %% Load data
all_subjects = [p for p in fmriprep_path.glob("sub-*") if p.is_dir()]
df_subj = {"framewise_displacement": [], "subject": [], "timepoint": []}
for subj_path in all_subjects:
    subj = subj_path.name
    confounds = pd.read_csv(
        subj_path
        / "func"
        / f"{subj}_task-{task}_desc-confounds_timeseries.tsv",
        sep="\t",
    )
    df_subj["framewise_displacement"] += confounds[
        "framewise_displacement"
    ].tolist()
    df_subj["subject"] += [subj] * len(confounds["framewise_displacement"])
    df_subj["timepoint"] += list(range(1, len(confounds) + 1))

df_all = pd.DataFrame(df_subj)
df_all = df_all.set_index(["subject", "timepoint"])
# Selects all rows starting from index 8 onwards for each subject group
df = df_all.groupby(level="subject", group_keys=False).apply(
    lambda x: x.iloc[8:]
)


# %% Descriptive analysis of outlier volumes in movement (>0.5)
percentage = (
    (df_all["framewise_displacement"] > 0.5).value_counts() / len(df_all) * 100
)
print(
    f"Framewise Displacement > 0.5 = {round(percentage.iloc[1], 2)} % of timepoints"
)

plt.hist(
    df["framewise_displacement"].dropna(), bins=100, color="cornflowerblue"
)

# Limit the viewable x-axis range (e.g., from 0 to 1.5 or 2.0 mm)
plt.xlim(0, 2)

plt.xlabel("Framewise Displacement (mm)")
plt.ylabel("Count of timepoints")
plt.title("Distribution of Framewise Displacement")
plt.text(
    0.23,
    45000,
    "X-axis truncated at 2 (one outlier at 16mm shrink the x-axis)",
)
plt.tight_layout()
plt.savefig(out_path / "framewise_displacement.png")
plt.close()
# %%
# %% Percentage of high-motion timepoints per subject
subj_percentage = (
    (df["framewise_displacement"] > 0.5).groupby(level="subject").mean()
    * 100  # Mean is % given average of 0 and 1
)
subj_percentage = subj_percentage.sort_values(ascending=False)

# Any subject with more than 20% of motion? (recomendation: to exclude)
print(
    f"Number of subjects with over 20% of movement outliers: {sum(subj_percentage > 20)}"
)


