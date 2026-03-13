# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

db_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/")

# %%
df = pd.read_csv(db_path / "participants.tsv", sep="\t")

# %%
# Gender distribution 
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Barplot for sex
sns.countplot(data=df, x="sex", ax=axes[0], color="lightblue")
axes[0].set_title("Gender Distribution")
axes[0].set_xlabel("Gender")
axes[0].set_ylabel("Count")

# Age distribution
##TODO: fix age, overwrite with age from questionnaire because here they all have +1
## years as for the MRI I use the date of the day of acquisition (to keep it anonymus, CRC policy)
## + year of participant birth (to accurately calculate metrics for MRI scanning, weight, etc)

sns.histplot( 
    data=df,
    x="age",
    kde=True,
    bins=10,
    ax=axes[1],
    color="lightblue",
    edgecolor="black",
)
axes[1].set_title("Age Distribution")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# %%
