# %%
# Documentation at: https://sites.google.com/site/bctnet/all-help-headers?authuser=0
import re
import pandas as pd
from junifer.storage import HDF5FeatureStorage
from pathlib import Path
import datatable as dt
from argparse import ArgumentParser


VALID_MARKERS = ["gsr", "noHighOrder", "noLowOrder", "noLowOrdernoAttLimb"]

SUMMARIES = {
    "DEGREESUND": ["std", "max", "skew"],
    "CLUSTERINGCOEFWU": ["mean", "std"],
    "BETWEENNESSWEI": ["mean", "std", "max"],
    "EIGENVECTORCENTRALITYUND": ["std", "max"],
}
KEEP_COLUMNS = {
    "STRENGTHSUNDSIGN": ["total_pos", "total_neg"],
}
NO_SUMMARY = {"EDGEBETWEENNESSWEI"} # same as betweenness

MAX_GLOBAL_COLS = 50 # Otherwise do summary (the idea is to capture ROI measures, could be much lower)

parser = ArgumentParser(
    description="Join graph metrics with events into one .jay per marker."
)

parser.add_argument(
    "--data",
    metavar="project_path",
    type=Path,
    help="Path to data",
    required=True,
)

parser.add_argument(
    "--marker",
    metavar="marker",
    type=str,
    choices=VALID_MARKERS,
    help=f"Subtype of IPC marker (folder), one of {VALID_MARKERS}. "
    "Default is IPC_all.",
    default=None,
)

parser.add_argument(
    "--p",
    type=float,
    default=None,
    help="Keep only thresholded metrics computed with this p "
    "(default: keep all variants found in the folder).",
)

parser.add_argument(
    "--klevel",
    type=int,
    default=None,
    help="Keep only rich_club_wu computed with this klevel "
    "(default: keep all variants found in the folder).",
)

args = parser.parse_args()

marker = args.marker if args.marker else ""
p_str = str(args.p).replace(".", "") if args.p is not None else None
k_str = str(args.klevel) if args.klevel is not None else None

data_path = args.data
graph_path = data_path / "junifer" / f"Graph_IPC{marker}"
events_path = data_path / "events"
out_path = data_path / "features"
out_path.mkdir(parents=True, exist_ok=True)

files = sorted(graph_path.glob("GRAPH*.hdf5"))
if not files:
    raise FileNotFoundError(f"No GRAPH*.hdf5 files found in {graph_path}")

frames = {}
sum_frames = {} 
for f in files:
    metricname = f.stem.removeprefix("GRAPH")  # e.g. degreesundp02, richclubwup02k10
    # Thresholded metrics end with p<digits> (and k<digits> for rich club)
    m = re.search(r"p(\d+)(?:k(\d+))?$", metricname)
    if m:
        if p_str is not None and m.group(1) != p_str:
            continue
        if k_str is not None and m.group(2) is not None and m.group(2) != k_str:
            continue
    metric = metricname.upper()
    raw = pd.read_hdf(f, key="df")
    df = raw.add_prefix(f"{metric}_")
    frames[metric] = df
    print(f"{metric}: {df.shape[1]} columns, {df.shape[0]} rows")

    # Summary version
    base = re.sub(r"P\d+(K\d+)?$", "", metric)  # DEGREESUNDP02 -> DEGREESUND
    if base in NO_SUMMARY:
        continue
    elif base in KEEP_COLUMNS:
        cols = [f"{metric}_{c}" for c in KEEP_COLUMNS[base]]
        sum_frames[metric] = df[cols]
    elif base in SUMMARIES:
        sum_frames[metric] = pd.DataFrame(
            {f"{metric}_{s}": getattr(raw, s)(axis=1) for s in SUMMARIES[base]}
        )
    elif df.shape[1] <= MAX_GLOBAL_COLS:
        sum_frames[metric] = df  # already global
    else:
        raise ValueError(
            f"{metric} has {df.shape[1]} columns and no summary defined: "
            "add it to SUMMARIES, KEEP_COLUMNS or NO_SUMMARY."
        )

# Join column-wise: every metric must have exactly the same rows
first_metric, first_df = next(iter(frames.items()))
for metric, df in frames.items():
    if not df.index.equals(first_df.index):
        raise ValueError(
            f"Index of {metric} differs from {first_metric}; "
            "recompute them with the same data / nan_policy."
        )

 
### Organizing IPC (filtering)
events = pd.read_csv(events_path / "all_events.csv")
events = events.set_index(["subject", "timepoint"])
outputs = {
    f"GRAPHipc{marker}": frames,
    f"GRAPHipc{marker}sum": sum_frames,
}
for name, metric_frames in outputs.items():
    df = pd.concat(metric_frames.values(), axis=1)
    if not df.columns.is_unique:
        raise ValueError(f"Duplicated column names in {name}.")
    print(f"{name}: joined {len(metric_frames)} metrics -> {df.shape}")
    print(f"{name}: joined {len(metric_frames)} metrics -> {df.shape}")
 
    # Keep only task-ES and join events (once)
    print("Selecting task-ES ...")
    df = df.xs("ES", level="task")
    print("Joining with events ...")
    df = df.join(events, how="inner")
    # Timepoints excluded (outside the inner joint):
    # (1) First times from 0 to 8 (until task starts)
    # (2) Event last timepoints calculated heuristically (stop recording)
 
    # Export to .jay
    print("Exporting to jay ...")
    df = df.reset_index()
    DT = dt.Frame(df)
    out_file = out_path / f"{name}.jay"
    DT.to_jay(str(out_file))
    print(f"Exported {df.shape} to {out_file}")
print("Done!")
# %%
