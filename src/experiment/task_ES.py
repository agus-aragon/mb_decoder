import logging 
from psychopy import visual, core, event, sound #, parallel
import psychotoolbox as ptb

class experience_sampling:
    """ A class to run a MB experience sampling task."""

    def __init__(self, params):

        # Task parameters
        self.subj             = params['subj']
        self.n_trials         = params['n_trials'] 
        self.interval         = params['interval']
        self.states           = params['states']
        self.task_length      = int((self.n_trials * self.interval) / 60)
        self.response_buttons = params['response_buttons']
        self.outfile         = f'sub_{self.subj}_task-ES.log'

        self.win = visual.Window(size  = (1270,720), 
                                 units = 'height', 
                                 color = 'white')
        self.clock  = core.Clock()
        self.logger = self.configure_logger()
        self.cross  = self.draw_cross()

        #parallel settings
        if params["parallel"]:
            self.parallel = True
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)

        print(f"Experiment created for subject {self.subj}")

    def configure_logger(self):
        """ Logger that prints messages to the console."""
        logger = logging.getLogger("ES_logger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        logger.addHandler(ch)

        logger.info(f'Logger configured for subject {self.subj}')
        return logger
    
    def draw_cross(self):
        """Fixation cross stimulus."""

        fixation_cross = visual.GratingStim(win=self.win,
                                   size=0.1, 
                                   pos=[0,0], 
                                   sf=0, 
                                   rgb=-1,
                                   mask='cross')
        return fixation_cross
    
    def play_probe(self, trial_num):
        """ Visual and auditory probe"""
        visual_probe = visual.TextStim(self.win,
                                       text='!',
                                       color='black',
                                       heigh=0.2,
                                       bold=True)

        auditory_probe = sound.Sound(value = 1000, 
                                     secs = 1.0,
                                     volume = 0.7)
        
        start_time = ptb.GetSecs()
        visual_probe.draw(when = start_time + 0.1)
        self.win.flip()
        auditory_probe.play(when = start_time + 0.1)
        
        self.logger.info(f"Probe {trial_num} at {start_time+0.1}")

        if self.parallel:
           self.port.setData(0x16)
           core.wait(0.01)
           self.port.setData(0) 
    


