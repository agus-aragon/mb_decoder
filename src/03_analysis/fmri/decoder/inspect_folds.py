#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    accuracy_score,
    recall_score,
    precision_score,
)

TR = 1.5
target = 'BlankvsMS'
cv = 'kfold'
# dimred = args.dimred (eventually)
window_start = -10
window_end = 0
model = 'logithc'
features = 'DISTANCES_distance' #'IPCnoLowOrder'

data_path = Path(f"/data/project/mb_decoder/output/03_analysis/decoder/cv_{cv}")
model_name = f"model-{model}_target-{target}_window-{window_start}-{window_end}_features-{features}_fold_predictions.csv"

df = pd.read_csv(data_path / model_name, sep=";", index_col=0)
# predict_repeat0_p0 = predicted label
# proba_repeat0_p0 = P(classes_[0])
# proba_repeat0_p1 = P(classes_[1])
# decision_repeat0_p0 = signed decision score
#       > 0 means the model predicts classes_[1]
#       < 0 means it predicts classes_[0]


# %%

trial_col = "trial_group"
y_col = "proba_repeat0_p1"#"predict_repeat0_p0"  # swap to "proba_repeat0_p1" for a smoother view


# 1-s bins over the window: [-10, -9), [-9, -8), ..., [-1, 0)
bin_edges = np.arange(window_start, window_end + 1, TR)


def binned_stats(data, y_col=y_col):
    """Mean and std of y_col per 1-s bin of seconds_to_probe.

    Timepoints are first averaged within each trial and bin, so every trial
    weighs the same in a bin no matter how many TRs it has there. With a 1.5 s
    TR a trial has at most one timepoint per bin, so the number of trials
    ("n") varies across bins.
    """
    t_df = data.assign(
        sec_bin=pd.cut(data["seconds_to_probe"], bin_edges, right=False)
    )
    per_trial = t_df.groupby(["sec_bin", trial_col], observed=True)[y_col].mean()
    stats = per_trial.groupby(level="sec_bin", observed=True).agg(
        ["mean", "std", "count"]
    )
    stats = stats.rename(columns={"count": "n"})
    stats["center"] = [b.mid for b in stats.index]
    return stats


def plot_binned(data, ax=None, y_col=y_col, color="darkblue", label=None, **kwargs):
    """Draw binned mean +/- std (kwargs absorbs FacetGrid's extra args)."""
    ax = ax or plt.gca()
    stats = binned_stats(data, y_col=y_col)
    ax.plot(
        stats["center"], stats["mean"], color=color, lw=2, marker="o", ms=4,
        label=label,
    )
    ax.fill_between(
        stats["center"],
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        color=color,
        alpha=0.15,
        lw=0,
    )


def plot_trials(df, target, y_col=y_col):
    """Prediction over the temporal window for trials with given target."""
    t_df = df[df["target"] == target].sort_values(
        [trial_col, "seconds_to_probe"]
    )

    # One line per trial, all subjects together
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(
        data=t_df,
        x="seconds_to_probe",
        y=y_col,
        units=trial_col,
        estimator=None,
        color="lightblue",
        alpha=0.5,
        lw=1,
        ax=ax,
    )
    plot_binned(t_df, ax=ax, y_col=y_col, label=f"mean ± SD ({TR}s bins)")
    ax.set_xlim(window_start, window_end)
    ax.set_xticks(range(window_start, window_end + 1))
    ax.legend()
    ax.set_title(
        f"target == {target}: {t_df[trial_col].nunique()} trials, "
        f"{t_df['subject'].nunique()} subjects"
    )
    plt.show()

    # One panel per subject, one line per trial
    g = sns.relplot(
        data=t_df,
        x="seconds_to_probe",
        y=y_col,
        col="subject",
        col_wrap=8,
        units=trial_col,
        estimator=None,
        kind="line",
        alpha=0.6,
        lw=1,
        height=1.8,
        aspect=1.2,
    )
    g.map_dataframe(plot_binned, y_col=y_col, color="darkblue")
    g.set(
        xlim=(window_start, window_end),
        xticks=range(window_start, window_end + 1, 2),
    )
    g.set_titles("{col_name}", size=12)
    g.figure.suptitle(f"target == {target}", y=1.01)
    plt.show()


# %%
# 1) Prediction over the window for target == 1 (Pos_label = MB **depending on the model)
plot_trials(df, target=1)

# %%
# 2) Prediction over the window for target == 0
plot_trials(df, target=0)

# %% 3) Trial summary of predictions
summary = df.groupby(trial_col).agg(
    subject=("subject", "first"),
    fold=("fold", "first"),
    target=("target", "first"),
    proba_repeat0_p0=("proba_repeat0_p0", "max"),
    proba_repeat0_p1=("proba_repeat0_p1", "max"),
    # any "1" predicted within the trial -> 1
    predict_repeat0_p0=("predict_repeat0_p0", "max"),
)

# %% 4) Trial-level metrics
y_true = summary["target"]
y_pred = summary["predict_repeat0_p0"]
metrics = {
    "auc": roc_auc_score(y_true, summary["proba_repeat0_p1"]),
    "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    "accuracy": accuracy_score(y_true, y_pred),
    "recall": recall_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred),
}
pd.Series(metrics)

# %%
# 5) AUC and balanced accuracy: timepoints (not summarized) vs trials (summary)
def split_scores(data, level):
    rows = []
    for split, t_df in data.groupby("fold"):
        rows.append(
            {
                "fold": split,
                "level": level,
                "auc": roc_auc_score(t_df["target"], t_df["proba_repeat0_p1"]),
                "balanced_accuracy": balanced_accuracy_score(
                    t_df["target"], t_df["predict_repeat0_p0"]
                ),
                "recall": recall_score(t_df["target"], t_df["predict_repeat0_p0"]),
                "precision": precision_score(t_df["target"], t_df["predict_repeat0_p0"]),
}
        )
    return pd.DataFrame(rows)


scores = pd.concat(
    [split_scores(df, "timepoint"), split_scores(summary, "trial")]
).melt(id_vars=["fold", "level"], var_name="metric", value_name="score")

#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

for ax, metrics, chance in [
    (ax1, ["auc", "balanced_accuracy"], 0.5),
    (ax2, ["recall", "precision"],      0.2),
]:
    subset = scores[scores["metric"].isin(metrics)]
    sns.boxplot(data=subset, x="metric", y="score", hue="level", ax=ax)
    sns.stripplot(
        data=subset, x="metric", y="score", hue="level", dodge=True,
        palette="dark:k", size=4, legend=False, ax=ax,
    )
    ax.axhline(chance, color="gray", ls="--", lw=1)
ax1.set_ylabel("score")
ax2.set_ylabel("")
plt.suptitle("One score per fold")
scores.groupby(["metric", "level"])["score"].describe()
# %%
