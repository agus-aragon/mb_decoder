# %% Initialization
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import normalize
from datatable import fread


centroids_path = Path(
    "/data/project/self_caught_mb/results/kmeans_SDC/corrected/rest-combined/cosine"
)
data_path = Path(
    "/data/project/mb_decoder/data/bids/mb_decoder/derivatives/features/"
)
IPC = fread(data_path / "IPC.jay")
IPC = IPC.to_pandas().set_index(["subject", "timepoint"])

kmeans = joblib.load(
    centroids_path / "kmeans_rest-combined-gs_k4_lag0s_cosine.pkl"
)


# %% ################################
# Distances
#####################################

# Distance calculation & data wrangling
fc_raw = IPC.drop(
    [
        "n_trial",
        "event",
        "seconds_to_probe",
        "response_prompt",
        "rt_prompt",
        "response_arousal",
        "rt_arousal",
    ],
    axis=1,
)
fc_z = fc_raw.apply(
    lambda x: np.arctanh(x.clip(-0.99999, 0.99999))
)  # Fisher Z-transform
# if distance == "cosine":
fc = normalize(fc_z, axis=1, norm="l2")  # L2 normalization (eucledian = cosine)
# else:
#     fc = fc_z.values

cluster_assignments = kmeans.predict(fc)
distances = kmeans.transform(fc)
#%%

# somatosensoy untegration
# dmn anticorrelation
# none to non
# hyperconnected
df_distances = pd.DataFrame(
    distances,
    columns=[f"dist_to_cluster_{i}" for i in range(distances.shape[1])],
)
df_distances["cluster_assigned"] = cluster_assignments
df_distances.index = IPC.index

df_distances["response"] = IPC["response"]
df_distances["response_merged"] = df_distances["response"].replace(
    {
        "future-other": "MW",
        "future-self": "MW",
        "present-sens": "Sens",
        "past-other": "MW",
        "past-self": "MW",
        "present-sdep": "MW",
    }
)
df_distances["response_merged"] = pd.Categorical(
    df_distances["response_merged"],
    categories=["scMB", "pMB", "MW", "Sens"],
    ordered=True,
)

event_id_combined = pd.concat([event_id_p, event_id_sc])
df_distances["event_id"] = event_id_combined
df_distances.to_csv(
    out_path / f"distance_.jay"  ##TODO
)
