# %%
import numpy as np
import pandas as pd
from pathlib import Path
from junifer.storage import HDF5FeatureStorage
import re
import datatable as dt


data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")
junifer_path = data_path / "junifer" 
events_path = data_path / "events"
out_path_events = data_path / "features"
out_path_events.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(events_path / "all_events.csv")
events = events.set_index(['subject', 'timepoint'])

masks = ['gm', 'wm', 'csf', 'brain']

# %% Load data
for mask in masks:
    GS_path = junifer_path / f"GS_{mask}"
    GS_file = HDF5FeatureStorage(uri=GS_path/ "GS_all.hdf5")
    GS_all = GS_file.read_df(f"BOLD_global_signal_{mask}_aggregation")
    # Keep only task-ES
    GS = GS_all.xs("ES", level="task")

    # Join events with IPC
    df = GS.join(events, how='inner') 
    # Timepoints excluded (outside the inner joint): 
        # (1) IPC first times from 0 to 8 (until task starts)
        # (2) Event last timepoints calculated heuristically (stop recording)

    # Export to .jay
    df = df.reset_index()
    DT = dt.Frame(df)
    DT.to_jay(str(out_path_events / f"GS{mask}.jay"))

# %%
