#%%
import numpy as np
import pandas as pd
from pathlib import Path
from junifer.storage import HDF5FeatureStorage
import re
import datatable as dt


def extract_network(region):
    # Schaefer-style: take only the first segment after hemi prefix as network
    m = re.match(r"^(LH|RH)_([A-Za-z]+)_", region)
    if m:
        return m.group(2)  # e.g. "Default", "DorsAttn", "Vis"

    # Tian-style: HIP-rh, pTHA-lh
    m = re.match(r"^[A-Za-z]+-(rh|lh)$", region)
    if m:
        return "Subcortex"

    raise ValueError(f"Unrecognized region pattern: {region}")

def new_name(col):
    a, b = col.split("~")
    net_a = extract_network(a)
    net_b = extract_network(b)

    if net_a == net_b:
        label = net_a.upper()
    else:
        label = f"INTERNETWORK_{net_a}_{net_b}"

    return f"{label}_{a}~{b}"


data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")
IPC_path = data_path / "junifer" / "IPC"
events_path = data_path / "events"
out_path_events = data_path / "features"
out_path_events.mkdir(parents=True, exist_ok=True)


# %% Load data
IPC_file = HDF5FeatureStorage(uri=IPC_path/ "IPC_all.hdf5")
IPC_all = IPC_file.read_df("BOLD_IPC_Schaefer_fc")

events = pd.read_csv(events_path / "all_events.csv")
events = events.set_index(['subject', 'timepoint'])

# %% Organize IPC

# Keep only task-ES
IPC = IPC_all.xs("ES", level="task")

# Keep only triangular matrix
seen = set()
keep_cols = []
for col in IPC.columns:
    a, b = col.split("~")
    key = tuple(sorted([a, b]))
    if key not in seen:
        seen.add(key)
        keep_cols.append(col)
IPC = IPC[keep_cols]

# Rename variables
IPC.columns = [new_name(c) for c in IPC.columns]

# Join events with IPC
df = IPC.join(events, how='inner') 
# Timepoints excluded (outside the inner joint): 
    # (1) IPC first times from 0 to 8 (until task starts)
    # (2) Event last timepoints calculated heuristically (stop recording)

# Export to .jay
df = df.reset_index()
DT = dt.Frame(df)
DT.to_jay(str(out_path_events / "IPC.jay"))
