# %%

# Hyphothesis 5: The frequency of MB behavioural reports will be positively
# associated with the MBQ score (trait-like MB) and the « Discontinuity of the Mind »
# scale of the ARSQ (experienced a restless mind during acquisition) and
# negatively associated with the « Self-consciousness » scale of the MCQ-30
# (tendency to constantly monitoring thoughts).

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as ticker

project_path = Path("/data/project/mb_decoder")
data_path = project_path / "data" / "bids" / "mb_decoder"
out_path = (
    project_path / "output" / "03_analysis" / "behavioral" / "questionnaires"
)
out_path.mkdir(parents=True, exist_ok=True)
rates_path = out_path.parent / "rates"
df = pd.read_csv(data_path / "participants.tsv", sep="\t")
rates = pd.read_csv(rates_path / "rates_task-ES.tsv",  sep="\t")

# Match rows by subject id instead of by row position
df = df.merge(rates, left_on="participant_id", right_on="subject", how="inner")

states = ["Blank", "Sleep", "Thought", "Sensation"]
colors = dict(
    Blank="#FFCA4F",
    Sleep="#EF4747",
    Thought="#1FA1CD",
    Sensation="#3DA336"
)

# %%
# Hyphothesis 5: The frequency of MB behavioural reports will be positively
# associated with the MBQ score (trait-like MB) and the « Discontinuity of the Mind »
# scale of the ARSQ (experienced a restless mind during acquisition) and
# negatively associated with the « Self-consciousness » scale of the MCQ-30
# (tendency to constantly monitoring thoughts).

# Change these to try other combinations (any participants.tsv column)
vars_of_interest = [
    "MBQ_score",
    "LMBS_score",
    "ARSQ_subscale_discontinuity_of_mind",
    'LMBS_subscale_experience_of_blank',
    'LMBS_subscale_memory_failure',
    'LMBS_subscale_stress_overflow',
    # "MCQ30_subscale_cognitive_selfconsciousness",
    # # 'hours_sleep_night_before',
    # 'stress_week_before',
    # 'stress_just_before_acquisition',
    # 'EHI_score',
    # 'MCQ30_score',
    # 'MCQ30_subscale_lack_of_cognitive_confidence',
    # 'MCQ30_subscale_positive_beliefs_about_worry',
    # 'MCQ30_subscale_cognitive_selfconsciousness',
    # 'MCQ30_subscale_negative_beliefs_about_uncontrollability_and_danger',
    # 'MCQ30_subscale_need_to_control_thoughts',
    # 'ACS_score',
    # 'ACS_subscale_focus',
    # 'ACS_subscale_shift',
    # 'ESS_score',
    # 'ARSQ_score',
    # 'ARSQ_subscale_discontinuity_of_mind',
    # 'ARSQ_subscale_theory_of_mind',
    # 'ARSQ_subscale_self',
    # 'ARSQ_subscale_planning',
    # 'ARSQ_subscale_sleepiness',
    # 'ARSQ_subscale_comfort',
    # 'ARSQ_subscale_somatic_awareness',
    # 'hour_acquisition',
    # 'confidence_thought',
    # 'confidence_blank',
    # # 'confidence_sleep',
    # 'confidence_sensations',
]
measure = "count"  # or "percentage"

# %% Statistics: Pearson and Kendall correlation for each state x variable
all_stats = []
for var in vars_of_interest:
    for state in states:
        col = f"{state.lower()}_{measure}"
        data = df[[var, col]]
        r, r_p = stats.spearmanr(data[var], data[col])
        tau, tau_p = stats.kendalltau(data[var], data[col])
        all_stats.append(dict(
            x=var, state=state, y=measure, n=len(data),
            r_spearman=r, p_spearman=r_p, tau_kendall=tau, p_kendall=tau_p,
        ))
stats_df = pd.DataFrame(all_stats)
# stats_df.to_csv(out_path / f"stats_{measure}.tsv", sep="\t", index=False)
stats_df

# %% Plot: one 2x2 figure per variable of interest, one panel per state
for var in vars_of_interest:
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, state in zip(axes.flat, states):
        col = f"{state.lower()}_{measure}"
        row = stats_df[
            (stats_df["x"] == var) & (stats_df["state"] == state)
        ].iloc[0]

        sns.regplot(
            data=df, x=var, y=col, ax=ax, color=colors[state],
            scatter_kws=dict(edgecolor="black", alpha=0.5),
        )
        r_text = f'{row.r_spearman:.2f}'.lstrip('0')
        p_text = f'{row.p_spearman:.3f}'.lstrip('0')
        k_text = f'{row.tau_kendall:.2f}'.lstrip('0')
        kp_text = f'{row.p_kendall:.3f}'.lstrip('0')
        ax.text(
            0.03, 0.97,
            f"r = {r_text}, p = {p_text}\n"
            f"τ = {k_text}, p = {kp_text}",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            color='black',
            bbox=dict(facecolor="white", edgecolor='black', alpha=0.6),

        )
        
        var_title = (var.replace("_"," ")).title()
        ax.set_title(state, fontweight="bold")
        ax.set_xlim(min(df[var])-1, max(df[var])+1)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
        ax.set_xlabel(var_title.upper())
        ax.set_ylabel(f"{state.upper()} {measure.upper()}")
    fig.suptitle(f"Correlation of {var_title} with task-ES reports", fontweight="bold")
    fig.tight_layout()
    # plt.savefig(out_path / f"scatter_{measure}_vs_{var}.png", dpi=300)
    plt.show()

# %%
