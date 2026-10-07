# %%

# Hyphothesis 5: The frequency of MB behavioural reports will be positively
# associated with the MBQ score (trait-like MB) and the « Discontinuity of the Mind »
# scale of the ARSQ (experienced a restless mind during acquisition) and
# negatively associated with the « Self-consciousness » scale of the MCQ-30
# (tendency to constantly monitoring thoughts).

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

project_path = Path("/data/project/mb_decoder")
data_path = project_path / "data" / "bids" / "mb_decoder"
out_path = (
    project_path / "output" / "03_analysis" / "behavioral" / "questionnaires"
)
rates_path = out_path.parent / "rates"
df = pd.read_csv(data_path / "participants.tsv", sep="\t")
rates = pd.read_csv(rates_path / "rates_task-ES.tsv",  sep="\t")

df = df.join(rates, how="inner")


# %%
# Hyphothesis 5: The frequency of MB behavioural reports will be positively
# associated with the MBQ score (trait-like MB) and the « Discontinuity of the Mind »
# scale of the ARSQ (experienced a restless mind during acquisition) and
# negatively associated with the « Self-consciousness » scale of the MCQ-30
# (tendency to constantly monitoring thoughts).


plt.scatter(df['stress_just_before_acquisition'], df['sleep_count'])

# %%
