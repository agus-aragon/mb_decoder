# %%
from julearn import run_cross_validation, PipelineCreator
from sklearn.model_selection import RepeatedStratifiedKFold
from julearn.stats.corrected_ttest import corrected_ttest
from julearn.viz import plot_scores
from seaborn import load_dataset
from datatable import fread
import numpy as np
import pandas as pd
from pathlib import Path


features = "IPC"
fname = f"{features}.jay"
project_path = Path("/data/project/mb_decoder/")
features_path = project_path / "data" / "bids"/ "mb_decoder" / "derivatives" / "features"
out_path_events = project_path / "output" / "03_analysis" / "decoder" / features
out_path_events.mkdir(parents=True, exist_ok=True)


df = fread(features_path / fname)
df = df.to_pandas().set_index(["subject", "timepoint"])

#%%
# TODO: define target, how big is the MS window, etc - new variable based on seconds to probe and response_prompt


## General pipeline for all models
X = [".+~.+"]
y = "target"
X_types={"DMN": ["DEFAULT_.*"],
         "VIS": ["VIS_.*"],
         "CONT": ["CONT_.*"],
         "DORSATTN":["DORSATTN_.*"],
         "LIMBIC": ["LIMBIC_.*"],
         "SALVENTATTN": ["SALVENTATTN_.*"],
         "SOMMOT": ["SOMMOT_.*"],
         "SUBCORTEX": ["SUBCORTEX_.*"],
         "INTERNETWORK": ["INTERNETWORK_.*"]
         }

creator = PipelineCreator(problem_type = "classification")
creator.add("zscore", with_mean=True)
creator.add("pca", n_components=0.2)
creator.add("svm")
cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
scoring = ["accuracy", "f1_macro"]


# Model SVM
search_params_svm = {
    "kind": "bayes", #could be other also cv for nested
    "n_iter": 10,
}

scores_svm = run_cross_validation(
    X = X,
    y = y,
    data = df,
    X_types = X_types,
    model=creator,
    cv=cv_splitter,
    scoring=scoring,
    return_train_score=True,
    return_estimator="all",
    search_params=search_params_svm,

)
print(scores_svm)
print(scores_svm.best_params_)
scores_svm["model"] = "svm"



# %% Statistical comparison of models 
# stats_df = corrected_ttest(scores, scores2, scores3)


panel = plot_scores(scores1, scores2, scores3)
panel.show()