from rest import resting_state 
from preparation import preparation
# from task_ES import experience_sampling
from task_ES_mac import experience_sampling  
from psychopy import event, core, visual


subject_id = '000'
parallel = False # Set to True when using EEG-fMRI

if __name__ == "__main__":
    # fMRI 1: Resting State
    params_rest = {
        "subj": subject_id,
        "duration": 1,  # total duration in minutes
        "parallel": parallel,
    }
    resting = resting_state(params_rest)
    resting.run_rest()
    
    # Filler to prepare for the next task
    wait = preparation()
    wait.run_preparation()

    
    # fMRI 2: Experience Sampling
    params_es = {
        "subj": subject_id,
        "n_trials": 50,
        "interval": 45,
        "jittering": 15,
        "duration": 40,  # total duration in minutes
        "states": ["Thought", "Blank", "Asleep"],
        "parallel": parallel,
        "response_buttons": ["b", "y", "g"],
    }
    experiment = experience_sampling(params_es)
    experiment.run_experiment()
    
    print("All tasks completed!")
