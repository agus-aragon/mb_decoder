# %%
# Documentation at: https://sites.google.com/site/bctnet/all-help-headers?authuser=0

import pandas as pd
from junifer.storage import HDF5FeatureStorage
from pathlib import Path
import datatable as dt
from argparse import ArgumentParser
from joblib import Parallel, delayed

from my_onthefly import read_transform
# from junifer.onthefly import read_transform

FEATURE_NAME = "BOLD_IPC_Schaefer_fc"
VALID_MARKERS = ["gsr", "noHighOrder", "noLowOrder", "noLowOrdernoAttLimb"]

# BCT functions supported by lib.read_transform
ALL_FUNCS = [
    "strengths_und_sign",
    "community_louvain",
    "degrees_und",
    "efficiency_wei",
    "strengths_und",
    "clustering_coef_wu",
    "eigenvector_centrality_und",
    "betweenness_wei",
    "edge_betweenness_wei",
    "distance_wei",
    "assortativity_wei",
    "transitivity_wu",
    "rich_club_wu",
]

# Functions that need p thresholding. Keep in sync with the `preprocessing`
# dict in lib/read_transform.py
THRESHOLDED = {
    "degrees_und",
    "efficiency_wei",
    "rich_club_wu",
    "transitivity_wu",
    "betweenness_wei",
    "edge_betweenness_wei",
    "distance_wei",
}

parser = ArgumentParser(
    description="Calculate graph metrics from IPC matrices."
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
    "--bctfuncs",
    metavar="FUNC",
    type=str,
    nargs="+",
    choices=ALL_FUNCS,  # althought it could be a not tested one.
    help="BCT functions to compute. Default: all defined",
    default=None,
)

# Optional parameters for read_transform
parser.add_argument(
    "--p",
    type=float,
    default=0.2,
    help="Proportion of strongest edges to keep. Used for thresholding "
    "(default: %(default)s).",
)

parser.add_argument(
    "--klevel",
    type=int,
    default=10,
    help="Maximum degree level for rich_club_wu (default: %(default)s).",
)

parser.add_argument(
    "--n_jobs",
    type=int,
    default=1,
    help="Number of functions computed in parallel, one process each "
    "(default: %(default)s). Each process loads the whole feature in "
    "memory.",
)

args = parser.parse_args()

# Validation of parameters
if not 0 < args.p <= 1:
    parser.error(f"--p must be in (0, 1], got {args.p}")
if args.klevel < 1:
    parser.error(f"--klevel must be >= 1, got {args.klevel}")
if args.n_jobs < 1:
    parser.error(f"--n_jobs must be >= 1, got {args.n_jobs}")

marker = args.marker if args.marker else ""
funcs = args.bctfuncs if args.bctfuncs else ALL_FUNCS
p = args.p
klevel = args.klevel
n_jobs = args.n_jobs
data_path = (
    args.data
)  # Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives")


ipc_path = data_path / "junifer" / f"IPC{marker}"
events_path = data_path / "events"
outpath = ipc_path.parent / f"Graph_IPC{marker}"

# Validation of paths
if not ipc_path.exists():
    raise FileNotFoundError(f"IPC path {ipc_path} does not exist.")
else:
    print(f"Path: {ipc_path}")
outpath.mkdir(parents=True, exist_ok=True)

if marker == "":
    file_suffix = "_all"
elif marker == "gsr":
    file_suffix = "gsr_all"
else:
    file_suffix = marker


def run_function(func):
    """Compute one BCT function and export it (one job per function)."""
    # Arguments for read_transform: each optional parameter is only passed
    # to the functions that use it

    transform_kw_args = {}
    preprocess_kw_args = {}

    if func in THRESHOLDED:
        preprocess_kw_args["p"] = p
    if func == "rich_club_wu":
        transform_kw_args["klevel"] = klevel
    func_str = str(func).replace("_", "")
    outname = f"GRAPH{func_str}"
    if func in THRESHOLDED:
        p_str = str(p).replace(
            ".", ""
        )  # Remove the dot from p to avoid issues in file names
        outname += f"p{p_str}"
    if func == "rich_club_wu":
        outname += f"k{klevel}"

    print(
        f"[{func}] computing "
        f"(transform_kw_args={transform_kw_args}, "
        f"preprocess_kw_args={preprocess_kw_args}) ..."
    )

    # Load data
    print(f"[{func}] loading events and IPC marker...")
    storage = HDF5FeatureStorage(uri=ipc_path / f"IPC{file_suffix}.hdf5")
    events = pd.read_csv(events_path / "all_events.csv")
    events = events.set_index(["subject", "timepoint"])

    print(f"[{func}] calculating...")

    df = read_transform(
        storage,
        feature_name=FEATURE_NAME,
        transform=f"bctpy_{func}",
        transform_kw_args=transform_kw_args,
        preprocess_kw_args=preprocess_kw_args,
    )
    print(f"[{func}] result shape: {df.shape}")
    print(f"[{func}] column names: {df.columns.tolist()[:10]} ...")

    ## Exported as junifer marker -> Junifer folder
    print(f"[{func}] exporting to HDF5 to {outpath}/{outname}.hdf5...")
    df.to_hdf(outpath / f"{outname}.hdf5", key="df", mode="w")

print(f"Computing BCT functions: {funcs} (n_jobs={n_jobs})")
Parallel(n_jobs=n_jobs)(delayed(run_function)(f) for f in funcs)
print("Done!")

# %%
