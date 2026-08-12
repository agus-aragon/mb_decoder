# %%
import re
from ast import Raise
import logging
from marshal import dump
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from datatable import fread
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    RepeatedKFold,
    train_test_split,
)
from sklearn.svm import LinearSVC

import julearn
from julearn import run_cross_validation
from julearn.config import set_config
from julearn.pipeline import PipelineCreator
from julearn.utils import LinearSVCHeuristicC, LogisticRegressionHeuristicC


# TODO: create logger
# TODO: comment after first run
set_config("disable_x_verbose", True)
set_config("disable_xtypes_verbose", True)
set_config("disable_xtypes_check", True)
set_config("disable_x_check", True)
# %%


################################################
# Argument parsing
################################################
start_time = time.time()

parser = ArgumentParser(description="Run the predictive models.")
parser.add_argument(
    "--target",
    metavar="target",
    type=str,
    help="Target to predict",
    required=True,
)
parser.add_argument(
    "--model",
    metavar="model",
    type=str,
    help="Model to use",
    required=True,
)
parser.add_argument(
    "--features",
    metavar="features",
    type=str,
    help="Features to use (METRIC_Xtypes (eg., IPC or IPC_DMN))",
    default=None,
    required=False,
)
parser.add_argument(
    "--optionals",
    metavar="optionals",
    type=str,
    help="optional features",
    required=False,
    default=None,
    nargs="*",
)

parser.add_argument(
    "--dimred",
    metavar="dimred",
    type=str,
    help="Optional dimensionality reduction method (eg., PCA)",
    required=False,
    default=None,
    nargs="*",
)
parser.add_argument(
    "--data", metavar="project_path", type=str, help="Path to data", default="../data"
)
parser.add_argument(
    "--fold",
    metavar="fold",
    type=int,
    help="Fold to compute. If None, compute all of them",
    default=None,
)
parser.add_argument(
    "--cv",
    metavar="cv",
    type=str,
    choices=["loso", "kfold", "nosplit"],
    help="Cross Validation strategy to compute (choices: %(choices)s)",
    default=None,
)

args = parser.parse_args()

N_REPEATS = 5
N_SPLITS = 5

target = args.target
model_name = args.model
features_name = args.features
fold = args.fold
cv = args.cv
dimred = [args.dimred] if args.dimred is not None else []

features_metric = features_name.split("_")[0]
features_xtypes = (
    features_name.split("_")[1:] if len(features_name.split("_")) > 1 else []
)


# %%
################################################
# Directories & Data
################################################
data_path = Path(args.data)
out_path = (
    data_path.parent
    / "output"
    / "03_analysis"
    / "decoder"
    / cv
    / f"target-{target}"
    / f"features-{features_metric}"
    / f"xtypes-{'_'.join(features_xtypes)}"
)
out_path.mkdir(parents=True, exist_ok=True)

df = fread(data_path / f"{features_metric}.jay")
df = df.to_pandas().set_index(["subject", "timepoint"])

features_suffix = ""
if features_xtypes is not None and len(features_xtypes) > 0:
    features_suffix = "_" + "-".join(features_xtypes)

dimred_suffix = ""
if dimred is not None:
    dimred_suffix = f"_{dimred}"

match = re.match(r"([a-zA-Z_]+)(\d+)", dimred)
if match:
    dimred_method, dimred_value = match.groups()
    dimred_value = int(dimred_value)
else:
    dimred_method = dimred
    dimred_value = None
if dimred_method in ["pca", "selectkbest"] and dimred_value is None:
    raise ValueError(
        f"Dimensionality reduction method '{dimred_method}' requires a value (e.g., 'pca95')."
    )


################################################
# Target Definition
################################################
filename = f"{model_name}_{features_metric}{features_suffix}{dimred_suffix}.joblib"
y = "target"
# TODO: define target, how big is the MS window, etc - new variable based on seconds to probe and response_prompt
# TODO: based on target argument can be defined diferently

# TODO
# agg parse targetdefinition:
    # window in seconds + seconds (seconds_10, seconds_5)
    # lag 
    # autodefined? by autocorrelation? by signal decoding?
    #  in another code + file, to be loaded here as definition TODO
    # yes and no, to not do circular validation 

################################################
# Feature Selection
################################################
if features_metric == "IPC":
    X_types = {
        "DMN": ["DEFAULT_.*"],
        "VIS": ["VIS_.*"],
        "CONT": ["CONT_.*"],
        "DORSATTN": ["DORSATTN_.*"],
        "LIMBIC": ["LIMBIC_.*"],
        "SALVENTATTN": ["SALVENTATTN_.*"],
        "SOMMOT": ["SOMMOT_.*"],
        "SUBCORTEX": ["SUBCORTEX_.*"],
        "INTERNETWORK": ["INTERNETWORK_.*"],
        "ONLYCORTICALNETWORKS": [
            "DEFAULT_.*",
            "VIS_.*",
            "CONT_.*",
            "DORSATTN_.*",
            "LIMBIC_.*",
            "SALVENTATTN_.*",
            "SOMMOT_.*",
        ],
        "ONLYNETWORKS": [
            "DEFAULT_.*",
            "VIS_.*",
            "CONT_.*",
            "DORSATTN_.*",
            "LIMBIC_.*",
            "SALVENTATTN_.*",
            "SOMMOT_.*",
            "SUBCORTEX_.*",
        ],
        "ALL": [".+~.+"],
    }
else:
    X_types = {"ALL": [".+"]}

if features_xtypes is not None and len(features_xtypes) > 0:
    X = features_xtypes
else:
    X = list(X_types.keys())

################################################
# General pipeline (applicable to any models)
################################################

creator = PipelineCreator(problem_type="classification", apply_to="*")
creator.add("zscore")

scoring = [
    "balanced_accuracy",
    "f1",
    "f1_macro",
    "average_precision",
    "roc_auc",
    "matthews_corrcoef",
    "recall",
    "recall_macro",
    "precision",
    "precision_macro",
]


################################################
# Feature dimensionality reduction (optional)
################################################
if "pca" in dimred_method:
    creator.add("pca", n_components=dimred_value)

if "selectkbest" in dimred_method:
    creator.add("SelectKBest", k=dimred_value)

if "cbpm" in dimred_method:
    creator.add(
        "cbpm",
        significance_threshold=0.05,
        corr_sign="posneg",
    )

elif "sfm_lasso" in dimred_method:
    selector = SelectFromModel(
        LinearSVC(penalty="l1", dual=False, C=0.01, class_weight="balanced")
    )
    creator.add(selector, name="select_from_model")


################################################
# Define model for binary clasification
################################################
search_params = None
predict_proba = "proba"

if model_name in ["rf", "et"]:
    creator.add(model_name, class_weight="balanced")

elif model_name == "svm":
    creator.add(model_name, probability=True, class_weight="balanced")

elif model_name == "gssvm":
    creator.add(
        "svm",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        kernel="linear",
        probability=True,
        class_weight="balanced",
    )
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}

elif model_name == "gsrf":
    n_estimators = [200, 500]
    criterion = ["gini", "entropy", "log_loss"]
    max_features = ["sqrt", "log2"]
    creator.add(
        "rf",
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
        n_jobs=1,
        class_weight="balanced",
    )
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}

elif model_name == "gset":
    n_estimators = [200, 500]
    max_features = ["sqrt", "log2"]
    criterion = ["gini", "entropy", "log_loss"]
    creator.add(
        "et",
        n_estimators=n_estimators,
        criterion=criterion,
        max_features=max_features,
        n_jobs=1,
    )
    search_params = {"kind": "grid", "scoring": "balanced_accuracy"}

elif model_name == "gslinearsvm":
    model = LinearSVC()
    creator.add(
        model,
        name="linearsvc",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        class_weight="balanced",
    )
    predict_proba = "decision"
    search_params = {
        "kind": "grid",
        "scoring": "balanced_accuracy",
        "pre_dispatch": "all",
    }

elif model_name == "linearsvm":
    model = LinearSVC()
    creator.add(
        model,
        name="linearsvc",
        C=0.001,
        dual=False,
        penalty="l1",
    )
    predict_proba = "decision"

elif model_name == "linearsvchc":
    model = LinearSVCHeuristicC()
    creator.add(
        model,
        name="linearsvcheuristicc",
        dual=False,
        penalty="l1",
    )
    n_jobs = 1
    predict_proba = "decision"

elif model_name == "logithc":
    model = LogisticRegressionHeuristicC()
    creator.add(
        model,
        name="logithc",
        dual=False,
        penalty="l1",
        solver="liblinear",
    )
    predict_proba = "decision"

elif model_name == "dummy":
    creator.add("dummy")

elif model_name == "dummy_stratified":
    creator.add("dummy", strategy="stratified")

elif model_name == "optunasvm_rbf":
    creator.add(
        "svm",
        C=(0.0001, 10000, "log-uniform"),
        kernel="rbf",
        gamma=(1e-7, 1000, "log-uniform"),
        probability=True,
    )
    search_params = {
        "kind": "optuna",
        "scoring": "balanced_accuracy",
        "n_trials": 50,
    }

elif model_name == "optunasvm":
    creator.add(
        "svm",
        C=(0.0001, 10000, "log-uniform"),
        kernel="linear",
        probability=True,
    )
    search_params = {
        "kind": "optuna",
        "scoring": "balanced_accuracy",
        "n_trials": 50,
    }

else:
    raise ValueError(
        f"Model '{model_name}' not recognized. Choose a valid model string."
    )


################################################
# Define CV
################################################

if cv == "loso":
    groups = df.index.get_level_values("subject")
    n_subjects = len(np.unique(groups))
    cv_splitter = StratifiedGroupKFold(
        n_splits=n_subjects
    )  # n_splits = number of unique subjects
elif cv == "kfold":
    groups = (
        df.index.get_level_values("subject").astype(str)
        + "_trial-"
        + df["n_trial"].astype(str)
    )
    cv_splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42
    )
elif cv == "nosplit":
    cv_splitter = None


################################################
# Run Model
################################################
model = run_cross_validation(
    X=X,
    y=y,
    data=df,
    X_types=X_types,
    model=creator,
    cv=cv_splitter,
    scoring=scoring,
    return_train_score=True,
    return_estimator="all",
    search_params=search_params,
)


################################################
# Export Model
################################################
model_path = out_path / filename
dump(model["estimator"], model_path)
