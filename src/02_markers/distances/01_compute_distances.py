# %% Initialization
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
import datatable as dt


centroids_path = Path(
    "/data/project/self_caught_mb/results/kmeans_SDC/corrected/rest-separate/cosine"#combined/cosine"
)
data_path = Path(
    "/data/project/mb_decoder/data/bids/mb_decoder/derivatives/features/"
)
IPC = dt.fread(data_path / "IPC.jay")
IPC = IPC.to_pandas().set_index(["subject", "timepoint"])

kmeans = joblib.load(
    centroids_path / "kmeans_rest-separate-p-gs_k4_lag0s_cosine.pkl" #"kmeans_rest-combined-gs_k4_lag0s_cosine.pkl"
)
#%%
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(float)
event_cols = [
    "n_trial",
    "event",
    "seconds_to_probe",
    "response_prompt",
    "rt_prompt",
    "response_arousal",
    "rt_arousal",
]

# Remove subcortical ROIs from IPC to match centroids from self-caught
subcortical_rois = {'HIP', 'AMY', 'pTHA', 'aTHA', 'NAc', 'GP', 'PUT', 'CAU'}

def get_both_sides(col):
    left, right = col.split('~')
    tokens = left.split('_')
    for i, t in enumerate(tokens):
        if t in {'LH', 'RH'}:
            left_clean = '_'.join(tokens[i:])
            return left_clean, right
    return None, right

def is_tian_roi(part):
    name = part.split('-')[0].split('_')[-1]
    return name in subcortical_rois

def should_keep(col):
    if '~' not in col:
        return True   # keep non-FC columns (n_trial, event, etc.)

    left, right = col.split('~')

    # Drop self-pairs (diagonal)
    left_clean, right_clean = get_both_sides(col)
    if left_clean == right_clean:
        return False

    # Drop if either side is a Tian ROI
    if is_tian_roi(right):
        return False
    left_roi = left.split('_')[-1]
    if '-' in left_roi and left_roi.split('-')[0] in subcortical_rois:
        return False

    # Drop pure subcortex-subcortex
    if col.startswith('SUBCORTEX_'):
        return False

    return True

keep_cols = [col for col in IPC.columns if should_keep(col)]
IPC_filtered = IPC[keep_cols].copy()

print(f"Keeping : {len(keep_cols)}")           # 4950 FC cols + 7 non-FC cols = 4957
print(f"Dropped : {len(IPC.columns) - len(keep_cols)}")
print(f"Shape   : {IPC_filtered.shape}")       # (88562, 4957)

# %% ################################
# Distances
#####################################

# Distance calculation & data wrangling
X = pd.read_parquet("/data/project/self_caught_mb/datasets/derivatives/features_SDC/fc/corrected/fc_filtered_p_0sPostProbe_gs.parquet")
X = X.drop(['response', 'event_id'], axis=1)

## Match order with self-caught ROIs
def to_canonical(col):
    if '~' not in col:
        return None
    left, right = col.split('~')
    tokens = left.split('_')
    for i, t in enumerate(tokens):
        if t in {'LH', 'RH'}:
            left_clean = '_'.join(tokens[i:])
            pair = tuple(sorted([left_clean, right]))
            return f"{pair[0]}~{pair[1]}"
    return None

# Build canonical -> original col mapping for IPC_filtered
ipc_fc_cols = [c for c in IPC_filtered.columns if c not in event_cols]
ipc_canonical_map = {to_canonical(c): c for c in ipc_fc_cols}

# Get X columns in order
x_fc_cols = [c for c in X.columns if '~' in c]

# Reorder IPC to match X column order
ordered_ipc_cols = []
missing = []
for x_col in x_fc_cols:
    canon = to_canonical(x_col)
    if canon in ipc_canonical_map:
        ordered_ipc_cols.append(ipc_canonical_map[canon])
    else:
        missing.append(x_col)

print(f"Matched  : {len(ordered_ipc_cols)}")  # 4950
print(f"Missing  : {len(missing)}")            #  0


fc_raw = IPC_filtered[ordered_ipc_cols] 

#%%
fc_z = fc_raw.apply(
    lambda x: np.arctanh(x.clip(-0.99999, 0.99999))
)  # Fisher Z-transform
# if distance == "cosine":
fc = normalize(
    fc_z, axis=1, norm="l2"
)  # L2 normalization (eucledian = cosine)
# else:
#     fc = fc_z.values
fc = fc.astype(np.float64)
distances = kmeans.transform(fc)
cluster_assignments = kmeans.predict(fc)

df_distances = pd.DataFrame(
    distances,
    columns=[
        "distance_visual",# "distance_somatosensory",
        "distance_dmn_anticorrelation",
        "distance_none_to_none",
        "distance_hyperconnectivity",
    ],
)
df_distances["cluster_assignments"] = cluster_assignments
df_distances.index = IPC_filtered.index

df_distances = df_distances.join(IPC_filtered[event_cols])

# %% Save to .jay
print("Exporting to jay...")
df = df_distances.reset_index()
DT = dt.Frame(df)
DT.to_jay(str(data_path / f"DISTANCES.jay"))
print("Done!")
