#%%
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

datapath = Path("/Users/agusaragon/Documents/psychopy")
subj = sys.argv[1]
datapath_subj = datapath / f"subj-{subj}"
# %%
task = yaml.load(datapath_subj / "task-ES" / f"sub_{subj}_task-ES_exp.yaml")




# %%
