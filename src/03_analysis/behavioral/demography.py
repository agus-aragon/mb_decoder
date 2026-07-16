## ## ################# Demography ################# ## ## 
# Describe sample demography (sex and age)               #
##########################################################

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

main_path = Path("/data/project/mb_decoder/")
db_path = main_path / "data" / "bids" / "mb_decoder"
out_path = main_path / "output" / "03_analysis" / "behavioral" / "demography"
out_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(db_path / "participants.tsv", sep="\t")

# %% Descriptive
age_mean = round(df['age'].describe(),2)
sex = df['sex'].value_counts()
porc_sex = round(sex / len(df) * 100,2)
sex_summary = pd.concat([sex, porc_sex], axis=1)
sex_summary.columns = ['N', 'porcentage']
print(f"Mean Age={int(age_mean['mean']):.2f} (SD={int(age_mean['std']):.2f})")
print(f"Female={int(sex_summary.loc['F', 'porcentage'])}% (N={int(sex_summary.loc['F', 'N'])})")
age_mean.to_csv(out_path / 'age.csv')
sex_summary.to_csv(out_path / 'sex.csv')

# %% Plot
# Gender distribution 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Barplot for sex
sns.countplot(data=df, x="sex", ax=axes[0], color="lightblue")
axes[0].set_title("Gender Distribution")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Count")

# Age distribution

sns.histplot(
    data=df,
    x="age",
    kde=True,
    bins=5,
    ax=axes[1],
    color="lightblue",
    edgecolor="black",
)
axes[1].set_title("Age Distribution")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    out_path / "demography.png",
    bbox_inches='tight', dpi=300
)
plt.show()


# %%
