# %%
# Documentation at: https://sites.google.com/site/bctnet/all-help-headers?authuser=0

import threading
import time
from contextlib import contextmanager
from datetime import timedelta
from junifer.storage import HDF5FeatureStorage
from pathlib import Path
from argparse import ArgumentParser
from joblib import Parallel, delayed
from sqlalchemy import func

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
    # "strengths_und",
    "clustering_coef_wu",
    "eigenvector_centrality_und",
    "betweenness_wei",
    # "edge_betweenness_wei",
    "distance_wei",
    "density_und_sign",
    "assortativity_wei",
    "transitivity_wu",
    # "rich_club_wu",
]

# Functions that need p thresholding. Keep in sync with the `preprocessing`
# dict in lib/read_transform.py
THRESHOLDED = {
    "degrees_und",
    "efficiency_wei",
    # "rich_club_wu",
    "transitivity_wu",
    "betweenness_wei",
    "edge_betweenness_wei",
    "distance_wei",
}
# Time helpers
HEARTBEAT_SECONDS = 300 # Every function prints a "still running" line this often (seconds)

def log(msg):
    """Print with a timestamp, flushed so it shows up in log files at once."""
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
 
 
def fmt_time(seconds):
    """Format seconds as H:MM:SS."""
    return str(timedelta(seconds=round(seconds)))
 
 
@contextmanager
def heartbeat(label, interval=HEARTBEAT_SECONDS):
    """Print a 'still running' line every `interval` s while the block runs."""
    stop = threading.Event()
    t0 = time.perf_counter()
 
    def beat():
        while not stop.wait(interval):
            log(f"[{label}] still running ... {fmt_time(time.perf_counter() - t0)} elapsed")
 
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
 

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
outpath = ipc_path.parent / f"Graph_IPC{marker}"

# Validation of paths
if not ipc_path.exists():
    raise FileNotFoundError(f"IPC path {ipc_path} does not exist.")
else:
    log(f"Path: {ipc_path}")
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
    t_start = time.perf_counter()

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

    log(
        f"[{func}] computing "
        f"(transform_kw_args={transform_kw_args}, "
        f"preprocess_kw_args={preprocess_kw_args}) ..."
    )

    # Load data
    log(f"[{func}] loading IPC {marker} marker...")
    storage = HDF5FeatureStorage(uri=ipc_path / f"IPC{file_suffix}.hdf5")

    log(f"[{func}] calculating...")
    with heartbeat(func):
        df = read_transform(
            storage,
            feature_name=FEATURE_NAME,
            transform=f"bctpy_{func}",
            transform_kw_args=transform_kw_args,
            preprocess_kw_args=preprocess_kw_args,
        )
    t_compute = time.perf_counter() - t_start

    log(f"[{func}] computed in {fmt_time(t_compute)} | shape: {df.shape}")
    log(f"[{func}] column names: {df.columns.tolist()[:10]} ...")

    ## Exported as junifer marker -> Junifer folder
    log(f"[{func}] exporting to HDF5 to {outpath}/{outname}.hdf5...")
    df.to_hdf(outpath / f"{outname}.hdf5", key="df", mode="w")
    elapsed = time.perf_counter() - t_start
    log(f"[{func}] done in {fmt_time(elapsed)}")
    return func, elapsed 
 
log(f"Computing BCT functions: {funcs} (n_jobs={n_jobs})")
t_total = time.perf_counter()
results = Parallel(n_jobs=n_jobs)(delayed(run_function)(f) for f in funcs)
total = time.perf_counter() - t_total
 
# Summary
log("Time per function (slowest first):")
for func, seconds in sorted(results, key=lambda r: r[1], reverse=True):
    log(f"  {func:28s} {fmt_time(seconds)}")
log(f"Sum of function times: {fmt_time(sum(s for _, s in results))}")
log(f"Total wall time:       {fmt_time(total)}")
log("Done!")
# %%
