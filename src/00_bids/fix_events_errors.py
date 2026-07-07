# %%
# ############################### Fix events errors ############################## #
## Some participants reported errors in their reports (eg., reporting sleep       ##
## when they felt a sensation). This script aims to fix those errors and          ##
## correct them according to the subject's feedback and report post acquisition.  ##
####################################################################################

import pandas as pd
import numpy as np
from pathlib import Path

data_path = Path("/data/project/mb_decoder/data/bids/mb_decoder")


# %% sub-003: Retrieve events from Psychopy
events_sub_003 = pd.read_csv(
    data_path / "sub-003" / "func" / "sub-003_task-ES_events.tsv", sep="\t"
)
#TODO: copy psychopy to Kronos     
# %% sub-005: Retrieve events from Psychop
events_sub_005 = pd.read_csv(
    data_path / "sub-005" / "func" / "sub-005_task-ES_events.tsv", sep="\t"
)


# %% sub-012: Invert arousal scale
events_sub_012 = pd.read_csv(
    data_path / "sub-012" / "func" / "sub-012_task-ES_events.tsv", sep="\t"
)
events_sub_012["response_arousal"] = 100 - events_sub_012["response_arousal"]
events_sub_012.to_csv(
    data_path / "sub-012" / "func" / "sub-012_task-ES_events.tsv",
    sep="\t",
    index=False,
)

# %% sub-031: Reported one sleep but it is not sleep
events_sub_031 = pd.read_csv(
    data_path / "sub-031" / "func" / "sub-031_task-ES_events.tsv", sep="\t"
)
sleep_idx_031 = events_sub_031.index[
    events_sub_031["response_mental_state"] == "Sleep"
]
events_sub_031.loc[sleep_idx_031, "response_mental_state"] = "Sensation"
events_sub_031.to_csv(
    data_path / "sub-031" / "func" / "sub-031_task-ES_events.tsv",
    sep="\t",
    index=False,
)


# %% sub-042: First sleep is a sensation. Sleep with aorusal 50% should be lower.
events_sub_042 = pd.read_csv(
    data_path / "sub-042" / "func" / "sub-042_task-ES_events.tsv", sep="\t"
)
sleep_idx_042 = events_sub_042.index[
    events_sub_042["response_mental_state"] == "Sleep"
]
events_sub_042.loc[sleep_idx_042[0], "response_mental_state"] = "Sensation"
events_sub_042.loc[sleep_idx_042]["response_arousal"] = events_sub_042.loc[
    sleep_idx_042
]["response_arousal"].where(
    events_sub_042.loc[sleep_idx_042]["response_arousal"] != 50, 30
) # Participant rated 30 in most sleep events.

events_sub_042.to_csv(
    data_path / "sub-042" / "func" / "sub-042_task-ES_events.tsv",
    sep="\t",
    index=False,
)
# %% sub-043:  First sleep arousal should be at 10%
events_sub_043 = pd.read_csv(
    data_path / "sub-043" / "func" / "sub-043_task-ES_events.tsv", sep="\t"
)
sleep_idx_043 = events_sub_043.index[
    events_sub_043["response_mental_state"] == "Sleep"
][0]
events_sub_043.loc[sleep_idx_043, "response_arousal"] = 10
events_sub_043.to_csv(
    data_path / "sub-043" / "func" / "sub-043_task-ES_events.tsv",
    sep="\t",
    index=False,
)


# %%
