###
# INPUT:
# 1. data_path = Path to the dataset containing features.
# 2. lags = List of time lags in seconds to process.
# 3. data_types = List of data types to process (e.g., 'all', 'rest').
#                 - 'all' includes both probe and self-caught + rest data (x2)
#                 - 'rest' includes only rest data (from each condition separate).
# 4. k = Number of clusters for KMeans.
# 5. excluded_subjects = List of subjects to exclude from the analysi
#
# DESCRIPTION:
# This script performs k-mean clustering on fc features. It can be performed
# on all data points (ppMB, scMB and rest) or only on rest data.
#
# OUTPUT:
# 1. model_kmeans_{prefix}_{data_type}_k{k}_{lag}sPostProbe.pkl: trained KMeans model
#    for each lag and data type, saved in the specified output directory.
###

# %% Initialization
import csv
import matplotlib.pyplot as plt
import joblib
from itertools import combinations
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from junifer.storage import HDF5FeatureStorage

# %% Define
data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives/junifer/IPC")
task = 'rest' #'rest' 'ES'
IPC_path = data_path / f"task-{task}"
distance = 'cosine'
k = 4
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

# %% Load fc data & Prepro
print("Loading fc data...")
IPC_file = HDF5FeatureStorage(IPC_path / f"IPC_task-{task}_all.hdf5")
IPC_raw = IPC_file.read_df("BOLD_IPC_Schaefer_fc")

with open((Path(__file__).parent / "utils" / "column_headers.csv"), newline='') as f:
    reader = csv.reader(f)
    roi_names = [row for row in reader][0]
roi_names = np.array(roi_names)[roi_order]
fc_columns = []
for a, b in combinations(roi_names, 2):
    fc_columns.append(f"{a}~{b}")
IPC_raw = IPC_raw[fc_columns]

# Fisher z-transform
IPC_z = IPC_raw.apply(lambda x: np.arctanh(x.clip(-0.99999, 0.99999)))  
if distance == "cosine":
    IPC = normalize(IPC_z, axis=1, norm="l2")  # L2 normalization (eucledian = cosine)
elif distance == "euclidean":
    IPC = IPC_z.values

#%% Kmeans
kmeans = KMeans(n_clusters=k, random_state=42, max_iter=200)
kmeans.fit(IPC)

#%% Plotting

# Default network boundaries (Schaefer 100x7)

network_bounds = {
        "DMN": (0, 24),
        "Cont": (24, 37),
        "SM": (37, 51),
        "Lm": (51, 57),
        "VA": (57, 68),
        "DA": (68, 84),
        "Vis": (84, 100),
}

n = 100  # Number of ROIs
x, y = np.triu_indices(n, k=1)

# Create figure
fig = plt.figure(figsize=(20, 6))
gs = fig.add_gridspec(1, k + 1, width_ratios=[1] * k + [0.05])
axs = [fig.add_subplot(gs[i]) for i in range(k)]
cbar_ax = fig.add_subplot(gs[k])

for cluster in range(k):
    # Create upper-triangle matrix
    matrix = np.zeros((n, n))
    matrix[x, y] = kmeans.cluster_centers_[cluster]
    mask = np.tri(n, k=-1)  # Mask lower triangle

    # Plot
    im = axs[cluster].imshow(
        np.ma.masked_where(mask, matrix), cmap="RdBu_r", vmin=-0.02, vmax=0.02
    )
    axs[cluster].set_title(f"Pattern {cluster+1}", fontsize=26)

    # Add network boundaries
    for net, (start, end) in network_bounds.items():
        # Vertical lines
        axs[cluster].plot(
            [start, start], [start, end], color="k", linewidth=1.5, clip_on=False
        )
        axs[cluster].plot(
            [end, end], [start, end], color="k", linewidth=1.5, clip_on=False
        )
        # Horizontal lines
        axs[cluster].plot(
            [start, end], [end, end], color="k", linewidth=1.5, clip_on=False
        )
        axs[cluster].plot(
            [start, end], [start, start], color="k", linewidth=1.5, clip_on=False
        )

        # Network labels
        axs[cluster].text(
            (start + end) / 2, end + 3, net, ha="center", va="top", fontsize=16
        )

    # Clean up axes
    axs[cluster].set_xticks([])
    axs[cluster].set_yticks([])
    axs[cluster].spines[:].set_visible(False)

# Add colorbar
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=12)
# cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
plt.tight_layout()
plt.show()

# %%
