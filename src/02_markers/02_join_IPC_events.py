#%%
from junifer.storage import HDF5FeatureStorage
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
import joblib
import csv

data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")
IPC_path = data_path / "IPC__"
events_path = data_path / "events"
out_path_events = data_path / "features"
out_path_events.mkdir(parents=True, exist_ok=True)


# %% Load data
IPC = HDF5FeatureStorage(IPC_path/ "IPC_all.hdf5")
IPC = IPC.read_df("BOLD_IPC_Schaefer_fc")

events = pd.read_csv(events_path / "all_events.csv")

# %% Organize IPC
# TODO: modify columns name network_ROI1~ROI2 or internetwork_ROI1~ROI2
# TODO: export a .jay
df = IPC.join(events, how='inner')


roi_order = np.array([
37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,

33, 34, 35, 36,
80, 81, 82, 83, 84, 85, 86, 87, 88,

9, 10, 11, 12, 13, 14,
58, 59, 60, 61, 62, 63, 64, 65,

30, 31, 32,
78, 79,

23, 24, 25, 26, 27, 28, 29,
73, 74, 75, 76, 77,

15, 16, 17, 18, 19, 20, 21, 22,
66, 67, 68, 69, 70, 71, 72,

0, 1, 2, 3, 4, 5, 6, 7, 8,
50, 51, 52, 53, 54, 55, 56, 57
])