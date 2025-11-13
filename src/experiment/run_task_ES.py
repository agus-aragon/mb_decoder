from task_ES import experience_sampling
# from task_ES_mac import experience_sampling  
from psychopy import event, core, visual


subject_id = '000'
parallel = True # Set to True when using EEG-fMRI

if __name__ == "__main__":
    # fMRI 2: Experience Sampling
    params_es = {
        "subj": subject_id,
        "n_trials": 50, #50
        "interval": 45,# 45
        "jittering": 15, #15
        "states": ["Thought", "Blank", "Asleep", "Sensation"],
        "parallel": parallel,
        "response_buttons": ["b", "y", "g", "r"],
    }
    experiment = experience_sampling(params_es)
    experiment.run_experiment()
    
print("ES task completed!")
