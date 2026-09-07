# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path("/data/project/mb_decoder/output/03_analysis/decoder/")
cv = "loso"
target = "MB10s"
features = ["GS-raw", "IPC-INTERNETWORK", "WM-raw"]
models = ["dummy_stratified", "linearsvm", "svm"]

def load_scores(
    features, cv, target, model, data_path
):
    feature_name, xtypes = features.split("-")
    folder = (
        data_path
        / cv
        / f"target-{target}"
        / f"features-{feature_name}"
        / f"xtypes-{xtypes}"
        / "folds"
        / model
    )
    files = sorted(folder.glob(f"{model}_*_scores.csv"))
    rows = []
    for f in files:
        df = pd.read_csv(f, sep=";", index_col=0)
        df['features'] = features
        df["model"] = model

        rows.append(df)

    return pd.concat(rows, ignore_index=True)


# %%
all_scores = pd.concat(
    [
        load_scores(feat, cv, target, model, data_path)
        for feat in features
        for model in models
    ],
    ignore_index=True,
)



long_scores = all_scores.melt(
    id_vars=["features", "model"],
    value_vars=["train_balanced_accuracy", "test_balanced_accuracy"],
    var_name="split",
    value_name="balanced_accuracy",
)
long_scores["split"] = long_scores["split"].str.replace("_balanced_accuracy", "")


# %%
# only plot test scores for the model-comparison view (train is useful separately for overfit checks)
plot_split = "test"
sub_scores = long_scores[long_scores["split"] == plot_split]

n_features = len(features)
n_models = len(models)
model_colors = dict(zip(models, plt.cm.tab10.colors))

fig, ax = plt.subplots(figsize=(2.2 * n_features + 2, 5))
rng = np.random.default_rng(0)
group_width = 0.8
model_width = group_width / n_models

for i, feat in enumerate(features):
    for j, model in enumerate(models):
        sub = sub_scores[(sub_scores["features"] == feat) & (sub_scores["model"] == model)]
        x_center = i + (j - (n_models - 1) / 2) * model_width
        jitter = rng.uniform(-model_width / 4, model_width / 4, size=len(sub))

        ax.scatter(
            x_center + jitter, sub["balanced_accuracy"],
            alpha=0.5, s=25, color=model_colors[model],
            label=model if i == 0 else None,
        )
        ax.scatter(
            x_center, sub["balanced_accuracy"].median(),
            marker="_", s=300, linewidths=3, color=model_colors[model], zorder=3,
        )

ax.set_xticks(range(n_features))
ax.set_xticklabels(features, rotation=15, ha="right")
ax.set_ylabel(f"{plot_split.capitalize()} balanced accuracy")
ax.set_title(f"Target {target} ({cv}) — model comparison")
ax.axhline(0.5, ls="--", color="grey", lw=1, alpha=0.7)
ax.legend(title="model", loc="upper right")
ax.set_ylim(0, 1)
fig.tight_layout()
plt.show()