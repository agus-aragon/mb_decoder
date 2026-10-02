#%%
import numpy as np
import pandas as pd
from pathlib import Path
from junifer.storage import HDF5FeatureStorage
from junifer.onthefly import read_transform

ipc_path = Path("/data/project/mb_decoder/data/bids/mb_decoder/derivatives/junifer/IPC")
storage = HDF5FeatureStorage(uri=ipc_path/"IPC_all.hdf5")
#%%
transformed_df = read_transform(
    storage,
    feature_name="BOLD_IPC_Schaefer_fc",
    transform="bctpy_degrees_und",
)



## Toolbox: 


# 1)
#  https://github.com/MRI-Lab-Graz/braingraph

# 2)
# https://github.com/fiuneuro/brainconn
# https://brainconn.readthedocs.io/en/latest/
# https://github.com/aestrivex/bctpy/tree/master


# 3) 
# https://github.com/wwu-mmll/photonai_graph
# %%
