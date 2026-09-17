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
    help="Model to use",
    required=False,
)

# valid_features = [  # TODO
#     "IPC",
#     "IPC_DEFAULT",
#     "IPC_VIS",
#     "IPC_CONT",
#     "IPC_DORSATTN",
#     "IPC_LIMBIC",
#     "IPC_SALVENTATTN",
#     "IPC_SOMMOT",
#     "IPC_SUBCORTEX",
#     "IPC_INTERNETWORK",
#     "IPC_ONLYCORTICALNETWORKS",
#     "IPC_ONLYNETWORKS",
#     "GSgm",
#     "GSgm_mean",
#     "GSgm_power",
#     "GSgm_derivative",
#     "GSwm",
#     "GSwm_mean",
#     "GSwm_power",
#     "GSwm_derivative",
#     "GScsf",
#     "GScsf_mean",
#     "GScsf_power",
#     "GScsf_derivative",
#     "GSbrain",
#     "GSbrain_mean",
#     "GSbrain_power",
#     "GSbrain_derivative",

# ]
# parser.add_argument(
#     "--feature",
#     metavar="feature",
#     type=str,
#     choices=valid_features,
#     nargs="+",
#     help="Features to use (METRIC_subtype (eg., IPC or IPC_DMN))",
#     required=False,
# )

# parser.add_argument(
#     "--subtype",
#     action="store_true",
#     help="Compare models with different subtypes of the same feature (eg., models for each subnetwork of IPC)",
# )

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
model = args.model if args.model is not None else None
# features_args = args.feature if args.feature is not None else None
cv = args.cv
data_path = args.data
# IS_SUBTYPE = args.subtype
dimred = args.dimred
IS_DISPLAY = args.display
IS_SAVE = args.save

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

filename_pattern = "model-"
filename_pattern += f"{model}_" if model is not None else "*"
filename_pattern += f"_target-{target_args}"
filename_pattern += f"_window-{window_start}-{window_end}"
filename_pattern += "_features-*"

# if IS_SUBTYPE:
#     for t_feature in features_args:
#         filename_pattern += rf"{t_feature}^_[^_]+$"
# elif features_args:
#     for t_feature in features_args:
#         filename_pattern += f"{t_feature}_"
# else:
#     filename_pattern += f"*_"

filename_pattern += "_scores.csv"
scores = {}
i = 0
for model_file in sorted(models_path.glob(filename_pattern)):
    i += 1
    if dimred == "no" and "_dimred-" in str(model_file):
        continue
    if dimred == "yes" and "_dimred-" not in str(model_file):
        continue
    print(f"Loading model {i}: {model_file}")
    df = pd.read_csv(model_file, sep=";", index_col=0)

    model_label = f"{df['model'].iloc[0]} | {df['features'].iloc[0]} | {df['dimred'].iloc[0]} "
    scores[model_file.stem] = df.assign(model=model_label, model_id=i)

panel = plot_scores(*scores.values())

filename = filename_pattern.replace("*", "ALL")
filename = filename.replace(".csv","")

if IS_SAVE:
    panel.save(out_path / f"{filename}.html")

if IS_DISPLAY:
    panel.show()

