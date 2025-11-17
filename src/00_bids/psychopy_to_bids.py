#%%
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

datapath = Path("/data/project/mb_decoder/data/subj_raw/") #Path("/Users/agusaragon/Documents/psychopy")
subj = '001' #sys.argv[1]
datapath_subj = datapath / f"sub-{subj}" / "psychopy"
# %%
with open(datapath_subj / "task-ES" / f"sub_{subj}_task-ES_exp.yaml", 'r') as file:
    task = pd.DataFrame(yaml.safe_load(file))



# %%
