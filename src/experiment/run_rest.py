from rest import resting_state 
from psychopy import event, core, visual


subject_id = '000'
parallel = False # Set to True when using EEG-fMRI

if __name__ == "__main__":
    # fMRI 1: Resting State
    params_rest = {
        "subj": subject_id,
        "parallel": parallel,
    }
    resting = resting_state(params_rest)
    resting.run_rest()
    

