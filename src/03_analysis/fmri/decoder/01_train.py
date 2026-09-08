# %%
import re
import logging
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
from sklearn.feature_selection import SelectFromModel
import julearn
from julearn import run_cross_validation
from julearn.config import set_config
from julearn.pipeline import PipelineCreator
from julearn.utils.logging import configure_logging, logger, raise_error

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "lib"))
from nimrls.ml import LinearSVCHeuristicC, LogisticRegressionHeuristicC

# set_config("disable_x_verbose", True)
# set_config("disable_xtypes_verbose", True)
# set_config("disable_xtypes_check", True)
# set_config("disable_x_check", True)


# %%
################################################
# Argument parsing
################################################
start_time = time.time()

parser = ArgumentParser(description="Run the predictive models.")

valid_targets = ["BlankvsMS", "BlankvsSleep", "ALLvsMS", "ALLvsSleep"]  # TODO
parser.add_argument(
    "--target",
    metavar="target",
    type=str,
    choices=valid_targets,
    help=(
        "Target state vs. MS or ALL + pre-probe window size in seconds "
        " (e.g., BlankvsMS Blank vs Mental States)",
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
    required=True,
)


valid_features = [  # TODO
    "IPC",
    "IPC_DMN",
    "IPC_VIS",
    "IPC_CONT",
    "IPC_DORSATTN",
    "IPC_LIMBIC",
    "IPC_SALVENTATTN",
    "IPC_SOMMOT",
    "IPC_SUBCORTEX",
    "IPC_INTERNETWORK",
    "IPC_ONLYCORTICALNETWORKS",
    "IPC_ONLYNETWORKS",
    "GS",
    "GS_raw",
    "GS_POWER",
    "GS_DERIVATIVE",
    "WM",
    "WM_raw",
    "WM_POWER",
    "WM_DERIVATIVE",
    "CSF",
    "CSF_raw",
    "CSF_POWER",
    "CSF_DERIVATIVE",
]
parser.add_argument(
    "--feature",
    metavar="feature",
    type=str,
    choices=valid_features,
    nargs="+",
    help="Features to use (METRIC_Xtypes (eg., IPC or IPC_DMN))",
    required=True,
)

valid_dimred = [
    "pca",
    "pca95",
    "pca90",
    "pca85",
    "pca80",
    "selectkbest10",
    "selectkbest20",
    "selectkbest50",
    "cbpm",
    "sfm_lasso",
]
parser.add_argument(
    "--dimred",
    metavar="dimred",
    type=str,
    choices=valid_dimred,
    help="Optional dimensionality reduction method (eg., pca95)",
    required=False,
    default=None,
    nargs="*",
)

parser.add_argument(
    "--data",
    metavar="project_path",
    type=Path,
    help="Path to data",
    required=True,
)

parser.add_argument(
    "--out-path",
    metavar="out-path",
    type=Path,
    help="Path to store results",
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
    "--debug",
    action="store_true",
    help="Run a fast sanity-check pass with less subjects.",
)

args = parser.parse_args()

N_REPEATS = 5
N_SPLITS = 5

DEBUG_N_SUBJECTS = 5
DEBUG_N_OPTUNA_TRIALS = 3

target_args = args.target
model_name = args.model
features_args = args.feature
cv = args.cv
dimred = args.dimred[0] if args.dimred is not None else None
data_path = args.data
out_path = args.out_path
IS_DEBUG_TEST = args.debug


# %%
################################################
# Directories & Data
################################################

window_start = args.window[0]
window_end = args.window[1]

features_suffix = "_".join(features_args)
dimred_suffix = f"_dimred-{dimred}" if dimred else ""

filename_path = f"cv_{cv}"
filename_suffix = (
    f"model-{model_name}_"
    f"target-{target_args}_"
    f"window-{window_start}-{window_end}_"
    f"features-{features_suffix}"
    f"{dimred_suffix}"
)


out_path = out_path / filename_path
out_path.mkdir(parents=True, exist_ok=True)

log_file = out_path / f"{filename_suffix}_log.log"
julearn.utils.configure_logging(level="INFO", fname=log_file, overwrite=False)

lh = logging.StreamHandler(sys.stdout)
output_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(fmt=output_format)
lh.setFormatter(formatter)  # set formatter
logger.setLevel("INFO")  # set level
logger.addHandler(lh)  # set handler
logger.info(f"Output filename path: {out_path}")
logger.info(f"Output filename suffix: {filename_suffix}")


## Validation of arguments
if not data_path.exists():
    raise_error(f"Data path '{data_path}' does not exist.")
if not data_path.is_dir():
    raise_error(f"Data path '{data_path}' is not a directory.")

logger.info(f"Target (args): {target_args}")

target_classes = target_args.split("vs")
pos_labels = target_classes[0]
logger.info(f"Target classes: {target_classes} | Positive label: {pos_labels}")

if window_start >= window_end:
    raise_error(
        f"Invalid window: start ({window_start}) must be less than end ({window_end})."
    )
else:
    logger.info(f"Window: {window_start} to {window_end} seconds")

logger.info(f"Model: {model_name}")
logger.info(f"Features: {features_args}")

logger.info(f"Dimensionality Reduction: {dimred if dimred else 'None'}")
logger.info(f"Cross-Validation: {cv if cv else 'None'}")


if IS_DEBUG_TEST:
    logger.warning(
        "Running in --debug mode: data is subsampled and outputs are "
        "prefixed with DEBUG_."
    )
##


## Read features
df = None
read_features = []
for this_feature in features_args:
    if "_" in this_feature:
        this_feature = this_feature.split("_")[0]
    if this_feature in read_features:
        continue
    logger.info(f"Loading feature: {this_feature}")
    t_df = fread(data_path / f"{this_feature}.jay")
    t_df = t_df.to_pandas().set_index(["subject", "timepoint"])
    col_rename = {
        col: f"{this_feature}_{col}"
        for col in t_df.columns
        if col
        not in [
            "n_trial",
            "event",
            "seconds_to_probe",
            "response_prompt",
            "rt_prompt",
            "response_arousal",
            "rt_arousal",
        ]
    }
    t_df = t_df.rename(columns=col_rename)
    if df is None:
        df = t_df
    else:
        columns_to_merge = [
            col for col in t_df.columns if col not in df.columns
        ]
        df = df.join(t_df[columns_to_merge], how="inner")
    read_features.append(this_feature)

# TODO: fix the logic so no matter which labels,
# only relevant subjects are kept.
has_target = (
    df["response_prompt"].eq("Blank").groupby(level="subject").transform("any")
)
df = df[has_target]

# Log info
logger.info(
    f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns, "
    f"{df.index.get_level_values('subject').nunique()} subjects"
)

kept_subjects = df.index.get_level_values("subject").unique()
logger.info(
    f"Keeping {len(kept_subjects)} subject(s) with target '{pos_labels}'"
)

if IS_DEBUG_TEST:
    rng = np.random.RandomState(42)
    all_subjects = df.index.get_level_values("subject").unique()
    debug_subjects = rng.choice(
        all_subjects, size=DEBUG_N_SUBJECTS, replace=False
    )
    df = df[df.index.get_level_values("subject").isin(debug_subjects)]
    logger.info(
        f"DEBUG: kept {DEBUG_N_SUBJECTS} subjects -> {df.shape[0]} rows"
    )
###################################################
# Data filtering based on window and target classes
###################################################
# Define length of events based on pre-defined window
logger.info(
    f"Filtering data based on window: {window_start} to {window_end} seconds"
)
window_mask = (df["seconds_to_probe"] >= window_start) & (
    df["seconds_to_probe"] <= window_end
)
df = df[window_mask]

logger.info(
    f"Filtered data: {df.shape[0]} rows, {df.shape[1]} columns, "
    f"{df.index.get_level_values('subject').nunique()} subjects"
)
# Define target based on the target categories
target_labels = []
for t_target in target_classes:
    if t_target == "Blank":
        target_labels.append("Blank")
    elif t_target == "Sleep":
        target_labels.append("Sleep")
    elif t_target == "Sensation":
        target_labels.append("Sensation")
    elif t_target == "Thought":
        target_labels.append("Thought")
    elif t_target == "MS":
        target_labels.append("Sensation")
        target_labels.append("Thought")
    elif t_target == "NoBlank":
        target_labels.append("Sensation")
        target_labels.append("Thought")
        target_labels.append("Sleep")

logger.info(f"Target labels: {target_labels}")
# Filter the dataframe to only include the target labels
df = df[df["response_prompt"].isin(target_labels)]

# Log info
counts = df.groupby(["subject", "n_trial"]).size()
logger.info(
    f"Target Window: {window_start}s - {window_end}s | Total Obs: {len(df)} | "
    f"Avg TRs/Trial: {counts.mean():.2f} | {counts.value_counts().to_dict()}"
)

target1_freq = df["response_prompt"].value_counts()[pos_labels]
target0_freq = df["response_prompt"].value_counts().sum() - target1_freq
logger.info(
    f"Target {pos_labels}: {target1_freq} observations "
    f"- {round(target1_freq / len(df) * 100, 2)}%"
    f" | Other {target_classes[1]}: {target0_freq} observations "
)

################################################
# Feature Selection
################################################
X = []
X_types = {}

for t_feature in features_args:
    t_subfeature = None
    if "_" in t_feature:
        feature_splits = t_feature.split("_")
        t_feature = feature_splits[0]
        t_subfeature = feature_splits[1]

    if t_subfeature is None:
        X.append(f"{t_feature}_.*")

        # TODO: add subfeature type logic here
        # For now, we just use the feature family as the type
        X_types[t_feature] = [f"{t_feature}_.*"]
    else:
        if t_subfeature in [
            "VIS",
            "CONT",
            "DORSATTN",
            "LIMBIC",
            "SALVENTATTN",
            "SOMMOT",
            "SUBCORTEX",
            "INTERNETWORK",
        ]:
            X.append(f"{t_feature}_{t_subfeature}_.*")
            if t_feature not in X_types:
                X_types[t_feature] = []
            X_types[f"{t_feature}"].extend([f"{t_feature}_{t_subfeature}_.*"])
        elif t_subfeature == "DMN":
            X.append(f"{t_feature}_DEFAULT_.*")
            if t_feature not in X_types:
                X_types[t_feature] = []
            X_types[f"{t_feature}"].extend([f"{t_feature}_DEFAULT_.*"])
        elif t_subfeature == "ONLYCORTICALNETWORKS":
            X.append(
                f"{t_feature}_(DEFAULT|VIS|CONT|DORSATTN|LIMBIC|SALVENTATTN|SOMMOT)_.*"
            )
            if t_feature not in X_types:
                X_types[t_feature] = []
            X_types[f"{t_feature}"].extend([
                    f"{t_feature}_(DEFAULT|VIS|CONT|DORSATTN|LIMBIC|SALVENTATTN|SOMMOT)_.*"
                ])
        elif t_subfeature == "ONLYNETWORKS":
            X.append(
                f"{t_feature}_(DEFAULT|VIS|CONT|DORSATTN|LIMBIC|SALVENTATTN|SOMMOT|SUBCORTEX)_.*"
            )
            if t_feature not in X_types:
                X_types[t_feature] = []
            X_types[f"{t_feature}"].extend([
                    f"{t_feature}_(DEFAULT|VIS|CONT|DORSATTN|LIMBIC|SALVENTATTN|SOMMOT|SUBCORTEX)_.*"
                ]
            )
        elif t_subfeature in ["raw", "POWER", "DERIVATIVE"]:
            if t_feature == "GS":
                t_feature_expand = "global_signal"
            elif t_feature == "WM":
                t_feature_expand = "white_matter"
            elif t_feature == "CSF":
                t_feature_expand = "csf"
            X.append(f"{t_feature_expand}_{t_subfeature.lower()}.*")
            if t_feature_expand not in X_types:
                X_types[t_feature_expand] = []
            X_types[f"{t_feature_expand}"].extend([
                    f"{t_feature_expand}_{t_subfeature.lower()}.*"
                ]
            )


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
# TODO: will this be done by X_types automatically?
# add here LEiDA?
if dimred == "pca":
    creator.add("pca")
elif dimred == "pca95":
    creator.add("pca", n_components=0.95)
elif dimred == "pca90":
    creator.add("pca", n_components=0.90)
elif dimred == "pca85":
    creator.add("pca", n_components=0.85)
elif dimred == "pca80":
    creator.add("pca", n_components=0.80)
elif dimred == "selectkbest10":
    creator.add("SelectKBest", k=10)
elif dimred == "selectkbest20":
    creator.add("SelectKBest", k=20)
elif dimred == "selectkbest50":
    creator.add("SelectKBest", k=50)
elif dimred == "cbpm":
    creator.add(
        "cbpm",
        significance_threshold=0.05,
        corr_sign="posneg",
    )
elif dimred == "sfm_lasso":
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

elif model_name == "gssvm":
    creator.add(
        "svm",
        C=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 1000000],
        kernel="linear",
        probability=True,
        class_weight="balanced",
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
    raise_error(
        f"Model '{model_name}' not recognized. Choose a valid model string."
    )

if (
    IS_DEBUG_TEST
    and search_params is not None
    and search_params.get("kind") == "optuna"
):
    logger.info(
        f"DEBUG: shrinking optuna n_trials {search_params['n_trials']} "
        f"-> {DEBUG_N_OPTUNA_TRIALS}"
    )
    search_params["n_trials"] = DEBUG_N_OPTUNA_TRIALS

################################################
# Define CV & Run Model
################################################
groups = None
groups_col = None
cv_splitter = None
if cv == "loso":
    df = df.reset_index()
    groups_col = "subject"
    groups = df[groups_col].values
    n_subjects = len(np.unique(groups))
    cv_splitter = StratifiedGroupKFold(
        n_splits=n_subjects, random_state=42, shuffle=True
    )
    logger.info(f"LOSO: {n_subjects} subjects -> {n_subjects} folds")

elif cv == "kfold":
    df = df.reset_index()
    trial_id = df['subject'].astype(str) + "_trial-" + df['n_trial'].astype(str)
    groups_col = "trial_group"
    df[groups_col] = trial_id
    cv_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=42
    )  # no premade function to do REPEATED stratified GROUP k fold
    for train_test in cv_splitter.split(df, df["response_prompt"], groups=df[groups_col]):
        train_idx, test_idx = train_test
        logger.info(
            f"KFold: {len(train_idx)} train rows | {len(test_idx)} test rows"
        )
        n_pos_train = (df.iloc[train_idx]["response_prompt"] == pos_labels).sum()
        n_pos_test = (df.iloc[test_idx]["response_prompt"] == pos_labels).sum()
        logger.info(
            f"Class balance: {n_pos_train} pos / {len(train_idx)} train | "
            f"{n_pos_test} pos / {len(test_idx)} test"
        )

################################################
# Select a single fold if --fold was given
################################################
return_estimator = "all"
if IS_DEBUG_TEST:
    filename_suffix = f"DEBUG_{filename_suffix}"

################################################
# Run Model
################################################
logger.info(f"Running cross-validation | cv={cv} | model={model_name}")
logger.info(
    f"Class balance going into CV: {df['response_prompt'].value_counts().to_dict()}"
)
out = run_cross_validation(
    X=X,
    y="response_prompt",
    data=df,
    X_types=X_types,
    pos_labels=pos_labels,
    model=creator,
    cv=cv_splitter,
    scoring=scoring,
    groups=groups_col,
    return_train_score=True,
    return_estimator="all",
    return_inspector=True,
    search_params=search_params,
)


################################################
# Export Model
################################################
scores, model, inspector = out

scores["cv_kind"] = cv
scores["model"] = model_name
scores["target"] = target_args
scores["window"] = f"{window_start}-{window_end}"
scores["features"] = features_suffix
scores["dimred"] = dimred if dimred else "None"

logger.info(f"Scores shape: {scores.shape}")
scores.to_csv(out_path / f"{filename_suffix}_scores.csv", sep=";")
joblib.dump(model, out_path / f"{filename_suffix}_model.joblib")

logger.info("Predicting fold probabilities")
try:
    if predict_proba == "proba":
        fold_predictions = inspector.folds.predict_proba()
    elif predict_proba == "decision":
        fold_predictions = inspector.folds.decision_function()
    else:
        fold_predictions = inspector.folds.predict()
    fold_predictions.to_csv(
        out_path / f"{filename_suffix}_fold_predictions.csv", sep=";"
    )
except Exception as e:
    logger.error(e)

elapsed_time = time.time() - start_time
logger.info(
    "Elapsed time {}".format(
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    )
)
logger.info("Done!")
