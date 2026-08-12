# %%
import numpy as np
import pandas as pd
from pathlib import Path
import re
import datatable as dt

data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")
fmriprep_path = data_path / "fmriprep"
events_path = data_path / "events"
out_path_events = data_path / "features"
out_path_events.mkdir(parents=True, exist_ok=True)


# %% Load data
all_subjects = [p for p in fmriprep_path.glob("sub-*") if p.is_dir()]

list_GS = []
list_CSF = []
list_WM = []

for subj_path in all_subjects:
    subj = subj_path.name
    df_subj = pd.read_csv(
        subj_path / "func" / f"{subj}_task-ES_desc-confounds_timeseries.tsv", sep="\t"
    )

    # 1. Global Signal DataFrame
    GS_cols = [
        "global_signal",
        "global_signal_derivative1",
        "global_signal_power2",
        "global_signal_derivative1_power2",
    ]
    GS_subj = df_subj[GS_cols].copy()
    GS_subj["subject"] = subj
    GS_subj['timepoint'] = GS_subj.index
    list_GS.append(GS_subj)

    # 2. CSF DataFrame
    CSF_cols = [
        "csf",
        "csf_derivative1",
        "csf_power2",
        "csf_derivative1_power2",
    ]
    CSF_subj = df_subj[CSF_cols].copy()
    CSF_subj["subject"] = subj
    CSF_subj['timepoint'] = CSF_subj.index
    list_CSF.append(CSF_subj)

    # 3. White Matter DataFrame
    WM_cols = [
        "white_matter",
        "white_matter_derivative1",
        "white_matter_power2",
        "white_matter_derivative1_power2",
    ]
    WM_subj = df_subj[WM_cols].copy()
    WM_subj["subject"] = subj
    WM_subj['timepoint'] = WM_subj.index
    list_WM.append(WM_subj)

# Combine all subject chunks into final DataFrames
GS_all = pd.concat(list_GS, ignore_index=True)
GS_all = GS_all.set_index(['subject', 'timepoint'])

CSF_all = pd.concat(list_CSF, ignore_index=True)
CSF_all = CSF_all.set_index(['subject', 'timepoint'])

WM_all = pd.concat(list_WM, ignore_index=True)
WM_all = WM_all.set_index(['subject', 'timepoint'])

#%%
events = pd.read_csv(events_path / "all_events.csv")
events = events.set_index(["subject", "timepoint"])


# Join events with IPC
df_GS = GS_all.join(events, how="inner")
df_CSF = CSF_all.join(events, how="inner")
df_WM = WM_all.join(events, how="inner")
# Timepoints excluded (outside the inner joint):
# (1) IPC first times from 0 to 8 (until task starts)
# (2) Event last timepoints calculated heuristically (stop recording)

# Check all DF have the same amount of timepoints
len(df_GS) == len(df_CSF) == len(df_WM)

#%% Export to .jay
df_GS = df_GS.reset_index()
DT_GS = dt.Frame(df_GS)
DT_GS.to_jay(str(out_path_events / "GS.jay"))

df_CSF = df_CSF.reset_index()
DT_CSF = dt.Frame(df_CSF)
DT_CSF.to_jay(str(out_path_events / "CSF.jay"))

df_WM = df_WM.reset_index()
DT_WM = dt.Frame(df_WM)
DT_WM.to_jay(str(out_path_events / "WM.jay"))