# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/")
n_probes = 50
colors = dict(
    Blank="#EEB42D",
    Sleep="#EF4747",
    Thought="#1FA1CD",
    Sensation="#5ECB57"
)

# %%
# Read all events.tsv files and concatenate them into a single DataFrame
all_events = []
for events_file in db_path.glob("**/func/*_task-ES_events.tsv"):
    print(f"Reading {events_file} ...   ")
    events_df = pd.read_csv(events_file, sep="\t")
    events_df["subject"] = (
        events_file.parent.parent.name
    )  # Extract subject from path
    all_events.append(events_df)
events_df = pd.concat(all_events, ignore_index=True)

# %% Frequency of each response per subject (% of out 50 probes)
response_counts = (
    events_df.groupby(["subject", "response_mental_state"])
    .size()
    .reset_index(name="count")
)
response_counts["percentage"] = response_counts["count"] / n_probes * 100
plt.figure(figsize=(12, 6))
sns.barplot(
    x="subject",
    y="percentage",
    hue="response_mental_state",
    palette=colors,
    data=response_counts,
)
plt.title("Percentage of Each Report per Subject", fontsize=16)
plt.xlabel("Subject", fontsize=14)
plt.ylabel("Percentage of Responses", fontsize=14)
plt.legend(title="Response", fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# %% Summary of Frequency per category
plt.figure(figsize=(6, 5))
sns.boxplot(
    x="response_mental_state",
    y="percentage",
    palette=colors,
    data=response_counts,
)
sns.scatterplot(
    x="response_mental_state",
    y="percentage",
    hue="response_mental_state",
    data=response_counts,
    palette=colors,
    s=50,
)
plt.legend().set_visible(False)
plt.title("Percentage of Each Report Across Subjects", fontsize=14)
plt.xlabel("Report", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.tight_layout()
plt.show()

# %%
