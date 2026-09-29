#%%
import numpy as np
import pandas as pd
from pathlib import Path
from junifer.storage import HDF5FeatureStorage
import re
import datatable as dt
from argparse import ArgumentParser


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

parser = ArgumentParser(description="Join IPC with events.")

parser.add_argument(
    "--data",
    metavar="project_path",
    type=Path,
    help="Path to data",
    required=True,
)
parser.add_argument(
    "--name",
    metavar="name",
    type=str,
    help="Feature name from HDF5.list_features() (eg., BOLD_IPC_Schaefer_fc)",
    required=True,
)

valid_markers = ['gsr', 'noHighOrder', 'noLowOrder', 'noLowOrdernoAttLimb' None]
parser.add_argument(
    "--marker",
    metavar="marker",
    type=str,
    choices=valid_markers,
    help="Subtype of IPC marker (folder)",
    required=False,
    default=None,
)

args = parser.parse_args()
name = args.name
marker = args.marker if args.marker else ""
data_path = args.data #Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")
ipc_path = data_path / "junifer" / f"IPC{marker}"
events_path = data_path / "events"
out_path_events = data_path / "features"
out_path_events.mkdir(parents=True, exist_ok=True)

print(f"Path: {ipc_path}")

# %% Load data
print(f"Loading IPC marker {name}...")
if marker == '':
    file_suffix = '_all'
elif marker == 'gsr':
    file_suffix = 'gsr_all'
else:
    file_suffix = marker
IPC_file = HDF5FeatureStorage(uri=ipc_path/f"IPC{file_suffix}.hdf5")
IPC_all = IPC_file.read_df(name)

events = pd.read_csv(events_path / "all_events.csv")
events = events.set_index(['subject', 'timepoint'])

### Organize IPC

# Keep only task-ES
print(f"Selecting task-ES IPC...")

IPC = IPC_all.xs("ES", level="task")

# Keep only triangular matrix
print("Keeping only triangular matrix...")
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
print("Renaming columns...")
IPC.columns = [new_name(c) for c in IPC.columns]

# Join events with IPC
print("Joining IPC with events ...")
df = IPC.join(events, how='inner') 
# Timepoints excluded (outside the inner joint): 
    # (1) IPC first times from 0 to 8 (until task starts)
    # (2) Event last timepoints calculated heuristically (stop recording)

# Export to .jay
print("Exporting to jay...")
df = df.reset_index()
DT = dt.Frame(df)
DT.to_jay(str(out_path_events / f"IPC{marker}.jay"))
print("Done!")
