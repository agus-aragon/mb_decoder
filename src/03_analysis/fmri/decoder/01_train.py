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

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "lib"))
from nimrls.ml import LinearSVCHeuristicC, LogisticRegressionHeuristicC
from nimrls.logging import (
    configure_logging,
    log_versions,
    logger,
    raise_error,
)

# set_config("disable_x_verbose", True)
# set_config("disable_xtypes_verbose", True)
# set_config("disable_xtypes_check", True)
# set_config("disable_x_check", True)

configure_logging()
julearn.utils.logging.configure_logging("INFO")
log_versions()


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
    help="Target state vs. MS or ALL + pre-probe window size in seconds (e.g., BlankvsMS10 for -10s to 0s)",
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
# parser.add_argument(
#     "--optionals",
#     metavar="optionals",
#     type=str,
#     help="optional features",
#     required=False,
#     default=None,
#     nargs="*",
# )

parser.add_argument(
    "--dimred",
    metavar="dimred",
    type=str,
    help="Optional dimensionality reduction method (eg., pca95)",
    required=False,
    default=None,
    nargs="*",
)
parser.add_argument(
    "--data",
    metavar="project_path",
    type=str,
    help="Path to data",
    default="../data",
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
features_args = args.features
fold = args.fold
cv = args.cv
dimred = args.dimred[0] if args.dimred is not None else None
# optionals = args.optionals if args.optionals is not None else []
IS_DEBUG_TEST = args.debug

if IS_DEBUG_TEST:
    logger.warning(
        "Running in --debug mode: data is subsampled and outputs are "
        "prefixed with DEBUG_."
    )

features_list = features_args.split("_")

target_match = re.match(r"([a-zA-Z_]+)(\d+)", target_args)
target_classes, target_window = target_match.groups()
target_classes = target_classes.split("vs")
pos_labels = target_classes[0]
target_window = int(target_window)

features_suffix = ""
if features_list is not None and len(features_list) > 0:
    features_suffix = "-" + "-".join(features_list)
# else:
#     features_suffix = "-ALL"

# optionals_suffix = ""
# if len(optionals) > 0:
#     optionals_suffix = "_" + "_".join(optionals)

dimred_suffix = ""
if dimred is not None:
    dimred_suffix = f"_{dimred}"
    dimred_match = re.match(r"([a-zA-Z_]+)(\d+)?", dimred)
    dimred_method, dimred_value = dimred_match.groups()
    dimred_value = int(dimred_value) if dimred_value else None
else:
    dimred_method = dimred
    dimred_value = None

if dimred_method in ["pca", "selectkbest"] and dimred_value is None:
    raise_error(
        f"Dimensionality reduction method '{dimred_method}' requires a value (e.g., 'pca95')."
    )

# %%
################################################
# Directories & Data
################################################
data_path = Path(args.data)
out_path = (
    data_path.parents[4]
    / "output"
    / "03_analysis"
    / "decoder"
    / cv
    / f"target-{target_args}s"
    / f"features-{features_suffix}"
)
out_path.mkdir(parents=True, exist_ok=True)

df = fread(data_path / f"{features_list[0]}.jay")
df = df.to_pandas().set_index(["subject", "timepoint"]).copy()

has_target = (
    df["response_prompt"]
    .eq(pos_labels)
    .groupby(level="subject")
    .transform("any")
)
df = df[has_target]

# Log info
logger.info(
    f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns, "
    f"{df.index.get_level_values('subject').nunique()} subjects"
)

kept_subjects = df.index.get_level_values("subject")[has_target].unique()
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

################################################
# Target Definition
################################################
# Define lenght of events based on pre-defined window
window_mask = (df["seconds_to_probe"] >= -target_window) & (
    df["seconds_to_probe"] <= 0.0
)
# Define target based on the target categories
if "ALL" in target_classes:
    df = df[window_mask].copy()
elif "MS" in target_classes:
    # Omit Sleep reports in response_prompt
    df = df[window_mask & (df["response_prompt"] != "Sleep")].copy()
elif "Sleep" in target_classes:
    df = df[
        window_mask
        & (df["response_prompt"] != "Sensation")
        & (df["response_prompt"] != "Thought")
    ].copy()

# Log info
counts = df.groupby(["subject", "n_trial"]).size()
logger.info(
    f"Target Window: {target_window}s | Total Obs: {len(df)} | "
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
if features_list is None or len(features_list) == 0:
    raise_error("No features specified. Use --features to specify features.")
elif features_list[0] not in ["IPC", "GS", "WM", "CSF"]:
    raise_error(
        f"Unknown feature metric '{features_list[0]}'. "
        "Valid options are: IPC, GS, WM, CSF."
    )
elif features_list[0] == "IPC":
    if "ALL" in features_list:
        X = df.filter(regex=".+~.+").columns.tolist()
    elif "DMN" in features_list:
        X = df.filter(regex="DEFAULT_.*").columns.tolist()
    elif "VIS" in features_list:
        X = df.filter(regex="VIS_.*").columns.tolist()
    elif "CONT" in features_list:
        X = df.filter(regex="CONT_.*").columns.tolist()
    elif "DORSATTN" in features_list:
        X = df.filter(regex="DORSATTN_.*").columns.tolist()
    elif "LIMBIC" in features_list:
        X = df.filter(regex="LIMBIC_.*").columns.tolist()
    elif "SALVENTATTN" in features_list:
        X = df.filter(regex="SALVENTATTN_.*").columns.tolist()
    elif "SOMMOT" in features_list:
        X = df.filter(regex="SOMMOT_.*").columns.tolist()
    elif "SUBCORTEX" in features_list:
        X = df.filter(regex="SUBCORTEX_.*").columns.tolist()
    elif "INTERNETWORK" in features_list:
        X = df.filter(regex="INTERNETWORK_.*").columns.tolist()
    elif "ONLYCORTICALNETWORKS" in features_list:
        X = df.filter(
            regex="DEFAULT_.*|VIS_.*|CONT_.*|DORSATTN_.*|LIMBIC_.*|SALVENTATTN_.*|SOMMOT_.*"
        ).columns.tolist()
    elif "ONLYNETWORKS" in features_list:
        X = df.filter(
            regex="DEFAULT_.*|VIS_.*|CONT_.*|DORSATTN_.*|LIMBIC_.*|SALVENTATTN_.*|SOMMOT_.*|SUBCORTEX_.*"
        ).columns.tolist()
elif features_list[0] == "GS":
    if "ALL" in features_list:
        X = df.filter(regex="global_signal.*").columns.tolist()
    elif "raw" in features_list:
        X = df.filter(regex="global_signal_raw").columns.tolist()
    elif "POWER" in features_list:
        X = df.filter(regex="global_signal_power.*").columns.tolist()
    elif "DERIVATIVE" in features_list:
        X = df.filter(regex="global_signal_derivative.*").columns.tolist()
elif features_list[0] == "WM":
    if "ALL" in features_list:
        X = df.filter(regex="white_matter.*").columns.tolist()
    elif "raw" in features_list:
        X = df.filter(regex="white_matter_raw").columns.tolist()
    elif "POWER" in features_list:
        X = df.filter(regex="white_matter_power.*").columns.tolist()
    elif "DERIVATIVE" in features_list:
        X = df.filter(regex="white_matter_derivative.*").columns.tolist()
elif features_list[0] == "CSF":
    if "ALL" in features_list:
        X = df.filter(regex="csf.*").columns.tolist()
    elif "raw" in features_list:
        X = df.filter(regex="csf_raw").columns.tolist()
    elif "POWER" in features_list:
        X = df.filter(regex="csf_power.*").columns.tolist()
    elif "DERIVATIVE" in features_list:
        X = df.filter(regex="csf_derivative.*").columns.tolist()

if features_list[0] == "IPC":
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
    }
elif features_list[0] == "GS":
    X_types = {
        "raw": ["global_signal_raw"],
        "POWER": ["global_signal_power.*"],
        "DERIVATIVE": ["global_signal_derivative.*"],
    }
elif features_list[0] == "WM":
    X_types = {
        "raw": ["white_matter_raw"],
        "POWER": ["white_matter_power.*"],
        "DERIVATIVE": ["white_matter_derivative.*"],
    }
elif features_list[0] == "CSF":
    X_types = {
        "raw": ["csf_raw"],
        "POWER": ["csf_power.*"],
        "DERIVATIVE": ["csf_derivative.*"],
    }
else:
    X_types = None


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
if dimred_method:
    if "pca" in dimred_method:
        pca_variance = dimred_value / 100
        creator.add("pca", n_components=pca_variance)
    elif "selectkbest" in dimred_method:
        creator.add("SelectKBest", k=dimred_value)
    elif "cbpm" in dimred_method:
        creator.add(
            "cbpm",
            significance_threshold=0.05,
            corr_sign="posneg",
        )
    elif "sfm_lasso" in dimred_method:
        selector = SelectFromModel(
            LinearSVC(
                penalty="l1", dual=False, C=0.01, class_weight="balanced"
            )
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
    cv_splitter = StratifiedGroupKFold(n_splits=n_subjects)
    logger.info(f"LOSO: {n_subjects} subjects -> {n_subjects} folds")

elif cv == "kfold":
    df = df.reset_index()
    trial_id = (
        df["subject"].astype(str) + "_trial-" + df["n_trial"].astype(str)
    )
    groups_col = "trial_group"
    df[groups_col] = trial_id
    groups = df[groups_col].values
    cv_splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=42
    )  # no premade function to do REPEATED stratified GROUP k fold
    logger.info(
        f"kfold: {df['trial_group'].nunique()} trial-groups -> {N_SPLITS} folds"
    )

################################################
# Select a single fold if --fold was given
################################################
return_estimator = "all"
if fold is not None:
    if cv_splitter is None:
        raise_error("Cannot select a single --fold when --cv is 'nosplit'.")
    all_folds = list(cv_splitter.split(df, df["response_prompt"], groups))
    cv_splitter = [all_folds[fold]]
    groups_col = None
    out_path = out_path / "folds" / model_name
    out_path.mkdir(parents=True, exist_ok=True)

    # if fold != 0:
    #     return_estimator = "cv"

suffix = f"_{fold}" if fold is not None else ""

filename = f"{model_name}{dimred_suffix}{suffix}"
if IS_DEBUG_TEST:
    filename = f"DEBUG_{filename}"

log_file = out_path / f"{filename}_log.log"
configure_logging(fname=log_file)
julearn.utils.configure_logging(level="INFO", fname=log_file, overwrite=False)

################################################
# Run Model
################################################
logger.info(
    f"Running cross-validation | cv={cv} | fold={fold} | model={model_name}"
)
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
    return_estimator=return_estimator,
    return_inspector=True,
    search_params=search_params,
    # seed?
)


################################################
# Export Model
################################################
scores, model, inspector = out
logger.info(f"Scores shape: {scores.shape}")
scores.to_csv(out_path / f"{filename}_scores.csv", sep=";")
joblib.dump(model, out_path / f"{filename}.joblib")

logger.info("Predicting fold probabilities")
try:
    if predict_proba == "proba":
        fold_predictions = inspector.folds.predict_proba()
    elif predict_proba == "decision":
        fold_predictions = inspector.folds.decision_function()
    else:
        fold_predictions = inspector.folds.predict()
    fold_predictions.to_csv(
        out_path / f"{filename}_fold_predictions.csv", sep=";"
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
