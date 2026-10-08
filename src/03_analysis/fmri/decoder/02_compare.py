# %%
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from julearn.viz import plot_scores
from argparse import ArgumentParser

parser = ArgumentParser(description="Compare models.")

valid_targets = ["BlankvsMS", "BlankvsSleep", "ALLvsMS", "ALLvsSleep"]
parser.add_argument(
    "--target",
    metavar="target",
    type=str,
    choices=valid_targets,
    help=(
        "Target state vs. MS or ALL (e.g., BlankvsMS Blank vs Mental States)",
    ),
    required=True,
)

parser.add_argument(
    "--window",
    metavar="window",
    type=int,
    help="Window of samples to use for each trial (in seconds).",
    required=True,
    nargs=2,
)

valid_models = [
    "rf",
    "et",
    "svm",
    "linearsvm",
    "gsrf",
    "gset",
    "gssvm",
    "gslinearsvm",
    "linearsvchc",
    "logithc",
    "dummy",
    "dummy_stratified",
    "optunasvm_rbf",
    "optunasvm",
]
parser.add_argument(
    "--model",
    metavar="model",
    type=str,
    choices=valid_models,
    nargs="+",
    help="Model to plot (default: all)",
    required=False,
)

parser.add_argument(
    "--features",
    metavar="features",
    type=str,
    nargs="+",
    help="Only runs whose features contain ANY of these values ANYWHERE "
    "(e.g. 'GS' also matches IPC_ONLYCORTICALNETWORKS_GSbrain). Default: all.",
    required=False,
    default=None,
)

parser.add_argument(
    "--dimred",
    metavar="dimred",
    type=str,
    choices=["yes", "no"],
    help="Include, exclude, or ignore dimensionality reduction models",
    required=False,
    default=None
)

parser.add_argument(
    "--data",
    metavar="project_path",
    type=Path,
    help="Path to data",
    required=True,
)


parser.add_argument(
    "--cv",
    metavar="cv",
    type=str,
    choices=["loso", "kfold", "nosplit"],
    help="Cross Validation strategy to compute (choices: %(choices)s)",
    required=True,
)

parser.add_argument(
    "--display",
    action="store_true",
    help="Open and show interactive html",
)

parser.add_argument(
    "--save",
    action="store_true",
    help="Save the interactive html",
)

args = parser.parse_args()

target_args = args.target
cv = args.cv
data_path = args.data
dimred = args.dimred
IS_DISPLAY = args.display
IS_SAVE = args.save
models = args.model
features = args.features

FAMILY_PREFIXES = ("IPC", "GRAPH", "GS", "DISTANCES")
MAX_SUB_LEN = 8
CUT_LEN = 4
METRIC_SEP = "+"
legend = {} 
 
def short_features(features):
    groups, family = {}, None
    for token in features.split("_"):
        if token.startswith(FAMILY_PREFIXES):
            family = token
            groups.setdefault(family, [])
        elif family is not None:
            groups[family].append(
                token if len(token) <= MAX_SUB_LEN else token[:CUT_LEN]
            )
    parts = [
        f"{fam}_{METRIC_SEP.join(subs)}" if subs else fam
        for fam, subs in groups.items()
    ]
    return METRIC_SEP.join(parts) or features
# %%
################################################
# Directories & Data
################################################
models_path = data_path / f"cv_{cv}"

window_start = args.window[0]
window_end = args.window[1]
window_str = f"{window_start}-{window_end}"

# features_suffix = "_".join(features_args) if features_args else None

out_path = data_path / "comparison"
out_path.mkdir(parents=True, exist_ok=True)

filename_pattern = "model-*"
filename_pattern += f"_target-{target_args}"
filename_pattern += f"_window-{window_start}-{window_end}"
filename_pattern += "_features-*"

filename_pattern += "_scores.csv"
scores = {}
i = 0


for model_file in sorted(models_path.glob(filename_pattern)):
    i += 1
    if dimred == "no" and "_dimred-" in str(model_file):
        continue
    if dimred == "yes" and "_dimred-" not in str(model_file):
        continue
    is_agg = "_agg-" in model_file.name
    agg_label = "trial" if is_agg else "TR"

    df = pd.read_csv(model_file, sep=";", index_col=0)
    full = df["features"].iloc[0]
    feats = base = short_features(full)
    n = 1
    while legend.get(feats, full) != full:  # same short label, DIFFERENT run
        n += 1  # (plot_scores would merge them into one group)
        feats = f"{base}#{n}"
    legend[feats] = full
    if models and df["model"].iloc[0] not in models:
        continue
    if features and not any(f in df["features"].iloc[0] for f in features):
        continue
    print(f"Loading model {i}: {model_file}")

    model_label = (
        f"{df['model'].iloc[0]} | {feats} | "
        f"{df['dimred'].iloc[0]} | {agg_label}"
    )
    scores[model_file.stem] = df.assign(model=model_label, model_id=i)

panel = plot_scores(*scores.values())

filename = filename_pattern.replace("*", "ALL")
filename = filename.replace(".csv","")
if models:
    filename += "_model-" + "-".join(models)
if features:
    filename += "_feature-" + "-".join(features)

if IS_SAVE:
    panel.save(out_path / f"{filename}.html")

if IS_DISPLAY:
    panel.show()

