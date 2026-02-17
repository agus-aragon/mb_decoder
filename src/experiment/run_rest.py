from rest import resting_state 
from psychopy import event, core, visual


subject_id = '000'
parallel = True # Set to True when using EEG-fMRI
duration = 10  # duration in minutes

if __name__ == "__main__":
    # fMRI 1: Resting State
    params_rest = {
        "subj": subject_id,
        "parallel": parallel,
        "duration": duration
    }
    resting = resting_state(params_rest)
    resting.run_rest()
    

