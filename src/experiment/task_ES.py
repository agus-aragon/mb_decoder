
import logging 
import random
from psychopy import visual, core, event, sound, parallel

class experience_sampling:
    """ A class to run a MB experience sampling task."""

    def __init__(self, params):

        # Task parameters
        self.subj             = params['subj']
        self.n_trials         = params['n_trials'] 
        self.interval         = params['interval']
        self.states           = params['states']
        self.parallel         = params["parallel"]
        self.task_length      = int((self.n_trials * self.interval) / 60)
        self.response_buttons = params['response_buttons']
        self.outfile          = f'sub_{self.subj}_task-ES.log'
        self.jittering        = params['jittering']

        self.win = visual.Window(size  = (1270,720), 
                                 units = 'height', 
                                 color = 'white')
        self.clock  = core.Clock()
        self.logger = self.configure_logger()
        self.cross  = self.draw_cross()

        #parallel settings
        if self.parallel:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)

        print(f"Experiment created for subject {self.subj}")

    def configure_logger(self):
        """ Logger that prints messages to the console."""
        logger = logging.getLogger("ES_logger")
        logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

        logger.info(f'Logger configured for subject {self.subj}')
        return logger
    
    def draw_cross(self):
        """Fixation cross stimulus."""

        fixation_cross = visual.GratingStim(win=self.win,
                                   size=0.1, 
                                   pos=[0,0], 
                                   sf=0, 
                                   color='black',
                                   mask='cross')
        return fixation_cross
    
    def play_probe(self, trial_num):
        #TODO: add auditory stimuus
        """ Visual and auditory probe"""
        visual_probe = visual.TextStim(self.win,
                                    text='!',
                                    color='black',
                                    height=0.2,
                                    bold=True)
        
        # auditory_probe = sound.Sound(value=1000,
        #                              secs=1,
        #                              volume=0.7)

        visual_probe.draw()
        self.win.flip()
        probe_time = self.clock.getTime()

        if self.parallel:
            self.win.callOnFlip(lambda: self.port.setData(0x16))

        # flip_time = self.win.flip()  # Returns exact time of visual onset
        self.logger.info(f"Probe {trial_num+1} at: {probe_time}")
        
        if self.parallel:
            core.wait(0.01)
            self.port.setData(0)

        core.wait(1.0)

        return probe_time

    def get_response(self, trial_num, probe_time):
        """Display response options and collect participant response."""
        self.win.flip()
        prompt_txt = ''
        for i, state in enumerate(self.states, 1):
            prompt_txt += f"{state}\n\n" 
        prompt = visual.TextStim(self.win,
                                 text    = prompt_txt,
                                 color   = 'black',
                                 height  = 0.09,
                                 wrapWidth = 1.8)
    
        prompt.draw()
        self.win.flip()
        # prompt_onset = self.clock.getTime()

        event.clearEvents()
        response = None

        if self.parallel:
            self.port.setData(0x20) 
            core.wait(0.01)
            self.port.setData(0)

        while response not in self.response_buttons:
            keys = event.getKeys(keyList=self.response_buttons + ['escape'], timeStamped=self.clock)
            if keys:
                key_pressed, key_time = keys[0]  
                if key_pressed == 'escape':
                    self.win.close()
                    core.quit()
                elif key_pressed:
                    response = key_pressed
                    response_time = key_time
                    rt = response_time - probe_time
                    state_index = int(response) - 1 
                    state_name = self.states[state_index]
                    self.logger.info(f"Trial {trial_num+1}: Response '{state_name}', (key {response}), RT: {rt:.3f} seconds, Raw time: {response_time:.3f}")
                    feedback_text = f"You chose: {state_name}"
                    feedback = visual.TextStim(self.win,
                                         text=feedback_text,
                                         color='black',
                                         height=0.1)
                    feedback.draw()
                    self.win.flip()
                    core.wait(1)
                    break
        
        if self.parallel:
            response_code = 0x30 + int(response)
            self.port.setData(response_code)
            core.wait(0.01)
            self.port.setData(0) 
    
        return response, rt, state_name

    def get_arousal_rating(self, trial_num):
        """Display arousal rating scale with continuous slider."""
        
        # Create scale elements
        question = visual.TextStim(self.win, 
                                text="How awake do you feel right now?",
                                color='black', 
                                height=0.07,
                                pos=(0, 0.3))
        
        left_label = visual.TextStim(self.win,
                                    text="Very sleepy",
                                    color='black',
                                    height=0.05,
                                    pos=(-0.4, 0.15))
        
        right_label = visual.TextStim(self.win,
                                    text="Very alert",
                                    color='black',
                                    height=0.05,
                                    pos=(0.4, 0.15))
        
        instruction = visual.TextStim(self.win,
                                    text="Use index and middle fingers to move, then press with the ring finger to confirm",
                                    color='black',
                                    height=0.04,
                                    pos=(0, -0.3))

        # Create slider
        slider_line = visual.Line(self.win,
                                start=(-0.4, 0),
                                end=(0.4, 0),
                                lineColor='black',
                                lineWidth=2)
        
        slider_marker = visual.Circle(self.win,
                                    radius=0.02,
                                    fillColor='red',
                                    lineColor='red',
                                    pos=(0, 0))

        if self.parallel:
            self.port.setData(0x40) 
            core.wait(0.01)
            self.port.setData(0)

        # Initialize rating and timing
        rating = 50 
        confirmed = False
        
        while not confirmed:
            # Update slider position
            x_pos = -0.4 + (rating / 100) * 0.8
            slider_marker.pos = (x_pos, 0)
            
            # Draw everything
            question.draw()
            left_label.draw()
            right_label.draw()
            instruction.draw()
            slider_line.draw()
            slider_marker.draw()
            prompt_onset = self.clock.getTime()

            # Show current value
            value_text = visual.TextStim(self.win,
                                        text=f"{rating}%",
                                        color='black',
                                        height=0.05,
                                        pos=(0, -0.1))
            value_text.draw()
            
            self.win.flip()
            
            # Check for key presses
            keys = event.getKeys(keyList=['1', '2', '3', 'escape'])
            
            if keys:
                if 'escape' in keys:
                    self.win.close()
                    core.quit()
                
                if '1' in keys:
                    rating = max(0, rating - 5)  # Decrease by 5%
                
                if '2' in keys:
                    rating = min(100, rating + 5)  # Increase by 5%
                
                if '3' in keys:
                    confirmed = True
                    rating_time = self.clock.getTime()
                    rt = rating_time - prompt_onset
                    break

        self.logger.info(f"Trial {trial_num+1}: Arousal rating {rating:.1f} (raw: {rating}), RT: {rt:.3f} seconds, Raw time: {rating_time:.3f} seconds")
        
        if self.parallel:
            rating_code = 0x40 + int(rating)
            self.port.setData(rating_code)
            core.wait(0.001)
            self.port.setData(0)

        self.win.flip()
        core.wait(0.1)
        return rating, rt

    def wait_scanner_trigger(self, n_triggers=5):
        """Wait for scanner trigger (e.g., '5' key press)."""
        self.logger.info(f"Waiting for {n_triggers} scanner triggers ...")
        trigger_times = []

        if self.parallel:
            self.logger.info("Waiting for parallel port triggers (0xFF)...")
            for i in range(n_triggers):
                self.logger.info(f"Waiting for trigger {i+1}/{n_triggers}...")
                while True:
                    port_data = self.port.readData()
                    if port_data == 0xFF:
                        trigger_time = self.clock.getTime()
                        trigger_times.append(trigger_time)
                        self.logger.info(f"Trigger {i+1} received at {trigger_time:.3f} seconds")
                        core.wait(0.1)
                        break
                    core.wait(0.0001)
        else:
            self.logger.info(f"Press 't' {n_triggers} times...")
            for i in range (n_triggers):
                self.logger.info(f"Waiting for trigger {i+1}/{n_triggers}...")
                keys = event.waitKeys(keyList=['t', 'escape'])
                if 'escape' in keys:
                    self.win.close()
                    core.quit()
                trigger_time = self.clock.getTime()
                trigger_times.append(trigger_time)
                self.logger.info(f"Trigger {i+1} received at {trigger_time:.6f} seconds")
        
        experiment_start_time = trigger_times[-1]
        self.logger.info(f"Experiment will start at last trigger: {experiment_start_time:.6f}")

        for i, time in enumerate(trigger_times):
            self.logger.info(f"Trigger {i+1}: {time:.6f} seconds")
        
        return experiment_start_time, trigger_time

    def run_trial(self, trial_num):
        """Run a single trial - handles all trial-specific operations."""
        trial_start = self.clock.getTime()
        self.logger.info(f"Trial {trial_num+1} started")
        
        # Show fixation cross for jittered interval
        self.cross.draw()
        self.win.flip()
        jitter = random.uniform(self.interval-self.jittering, self.interval+self.jittering)
        core.wait(jitter)
        
        # Play the probe
        probe_time = self.play_probe(trial_num)
        
        # Get response
        response, rt, state = self.get_response(trial_num, probe_time)
        
        # Get arousal rating only for Mind Blanking
        arousal, arousal_rt = None, None
        if state == "Mind Blanking":
            arousal, arousal_rt = self.get_arousal_rating(trial_num)
        
        # Return trial data (you might want to save this to file)
        return {
            'trial_num': trial_num + 1,
            'state': state,
            'response_time': rt,
            'arousal': arousal,
            'arousal_rt': arousal_rt,
            'probe_time': probe_time
        }

    def save_log(self):
        #TODO: save log

    def run_experiment(self):
        """Main experiment loop - coordinates the overall flow."""
        # Wait for scanner triggers
        self.experiment_start_time, self.trigger_times = self.wait_scanner_trigger(n_triggers=5)
        self.logger.info(f"Experiment synchronized to last trigger at: {self.experiment_start_time:.6f}")
        
        # Run all trials
        for trial_num in range(self.n_trials):
            trial_data = self.run_trial(trial_num)
            
            # Log trial completion with timing info
            trial_end = self.clock.getTime() - self.experiment_start_time
            self.logger.info(f"Trial {trial_num+1} completed at {trial_end:.3f}s after sync")

        self.logger.info("Experiment completed successfully!")
        self.win.close()

################################
################################
# Example usage:
################################
################################

if __name__ == "__main__":
    params = {
        'subj': '001',
        'n_trials': 5,
        'interval': 5,  # seconds between trials
        'jittering':2,
        'states': ['Thought', 'Mind Blanking', 'Asleep'],
        'parallel': False,
        'response_buttons': ['1', '2', '3']
    }
    
    experiment = experience_sampling(params)
    experiment.run_experiment()