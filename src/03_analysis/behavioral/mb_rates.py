## ## ##################### MB Rates ##################### ## ##
# #
# #
# #
# #
################################################################
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

main_path = Path("/data/project/mb_decoder/")
db_path = main_path / "data" / "bids" / "mb_decoder"
out_path = main_path / "output" / "03_analysis" / "behavioral" / "mb_rates"
out_path.mkdir(parents=True, exist_ok=True)

n_probes = 50
colors = dict(
    Blank="#EEB42D", Sleep="#EF4747", Thought="#1FA1CD", Sensation="#5ECB57"
)

# %%
# Read all events.tsv files and concatenate them into a single DataFrame
all_events = []
for events_file in db_path.glob("**/func/*_task-ES_events.tsv"):
    print(f"Reading {events_file} ...   ")
    event_subj_df = pd.read_csv(events_file, sep="\t")
    event_subj_df["subject"] = (
        events_file.parent.parent.name
    )  # Extract subject from path
    all_events.append(event_subj_df)
events_df = pd.concat(all_events, ignore_index=True)

# %% Frequency of each response per subject (% of out 50 probes)
states = ["Blank", "Sleep", "Thought", "Sensation"]

response_counts = (
    events_df.groupby(["subject", "response_mental_state"])
    .size()
    .reset_index(name="count")
)

all_pairs = pd.MultiIndex.from_product(
    [events_df["subject"].unique(), states],
    names=["subject", "response_mental_state"],
)

response_counts = (
    response_counts.set_index(["subject", "response_mental_state"])
    .reindex(all_pairs, fill_value=0)
    .reset_index()
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
plt.savefig(out_path / "ms_rate_per_participant.png")
plt.tight_layout()
plt.show()


# %% Summary of Frequency per category
plt.figure(figsize=(5, 5))
sns.boxplot(
    x="response_mental_state",
    y="percentage",
    palette=colors,
    data=response_counts,
)
sns.stripplot(
    x="response_mental_state",
    y="percentage",
    hue="response_mental_state",
    data=response_counts,
    palette=colors,
    s=4,
    edgecolor="black",
    linewidth=1,
    jitter=0.1,
)
plt.legend().set_visible(False)
plt.title("Percentage of Each Report Across Subjects", fontsize=14)
plt.xlabel("Report", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.tight_layout()
plt.savefig(out_path / "ms_rate.png")
plt.show()

response_counts.to_csv(out_path / "response_counts.csv")

# %%
subj_no_mb_reports = response_counts[
    (response_counts["count"] == 0)
    & (response_counts["response_mental_state"] == "Blank")
]
subj_mb_reports = response_counts[
    (response_counts["count"] != 0)
    & (response_counts["response_mental_state"] == "Blank")
]
len_no_mb = len(subj_no_mb_reports)
len_mb = len(subj_mb_reports)
counts = [len_no_mb, len_mb]
categories = ["No MB Reports", "MB Reports"]
total = sum(counts)


bar_labels = [
    f"{count}\n({count / total * 100:.1f}%)" if total > 0 else "0 (0%)"
    for count in counts
]
plt.figure(figsize=(5, 4))
ax = sns.barplot(
    x=categories,
    y=counts,
    color="lightblue",
    edgecolor="black",
)
ax.bar_label(ax.containers[0], labels=bar_labels, padding=3)

plt.ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
plt.ylabel("Subject Count")
plt.title("MB Reports")
plt.savefig(out_path / "subj_no_mb_reports.png")
plt.show()


# %%
response_counts['MS_flag'] = np.where(
    response_counts['response_mental_state'].astype(str).str.contains('Blank', case=False, na=False),
    'Blank',
    'MS'
)

# 2. Compress counts: group by subject and MS_flag, then sum counts
compressed_counts = response_counts.groupby(['subject', 'MS_flag'])['count'].sum().unstack(fill_value=0)

# 3. Convert counts to proportions (0.0 to 1.0)
prop_df = compressed_counts.div(compressed_counts.sum(axis=1), axis=0)

# 4. Plot 100% stacked bar chart
fig, ax = plt.subplots(figsize=(10, 5))

prop_df[['MS', 'Blank']].plot(
    kind='bar',
    stacked=True, 
    color=['navy', 'gold'], 
    ax=ax,
    width=0.6
)

# Formatting
ax.legend(['MS', 'Blank'], bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
ax.axhline(0.5, color='black', linestyle='--', linewidth=1)  # 50% chance line
ax.set_ylabel('Sample Proportion')
ax.set_xlabel('Participants')
ax.set_ylim(-0.02, 1.02)
plt.xticks(rotation=90, ha='right')

plt.tight_layout()
plt.savefig(out_path / "proportion_MB_vs_MS.png")
plt.show()
# %%
