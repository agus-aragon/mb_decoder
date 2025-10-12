# from psychopy import prefs
# prefs.hardware['audioDevice'] = 'SPDIF-Out (SB Recon3D PCIe)'
# import psychtoolbox.audio
import yaml
import logging
import random
from pathlib import Path
import shutil
# from psychtoolbox import audio
from psychopy import visual, core, event, parallel, sound
# from psychopy.sound import backend_ptb

SCANNER = 15
EXPERIMENT = 25
SCANNER_SUMMARY = 26
logging.addLevelName(SCANNER, "SCANNER")
logging.addLevelName(SCANNER_SUMMARY, "SCANNER_SUMMARY")
logging.addLevelName(EXPERIMENT, "EXPERIMENT")
picture_dir = Path(__file__).parent / "pictures"


class experience_sampling:
    """A class to run a MB experience sampling task."""

    def __init__(self, params):
        # Task parameters
        self.subj = params["subj"]
        self.n_trials = params["n_trials"]
        self.interval = params["interval"]
        self.states = params["states"]
        self.parallel = params["parallel"]
        self.duration = params["duration"]
        self.response_buttons = params["response_buttons"]
        self.exp_dir = Path(__file__).parent / f"sub-{self.subj}" /  "task-ES"
        if self.exp_dir.exists():                                                 
            raise ValueError("An experiment dir for this subject already exists") 
        self.exp_dir.mkdir(parents=True)               
        self.logfile = self.exp_dir / f"sub_{self.subj}_task-ES_log.log"
        self.eventfile = self.exp_dir / f"sub_{self.subj}_task-ES_ev.yaml"
        self.expfile = self.exp_dir / f"sub_{self.subj}_task-ES_exp.yaml"
        self.jittering = params["jittering"]

        self._volume_count = 0
        self._events = []

        self.win = visual.Window(size=(1010, 520), units="height", color="white")
        self.clock = core.Clock()
        self.logger = self.configure_logger()
        self.cross = self.draw_cross()

        # parallel settings
        if self.parallel:
            self.port = parallel.ParallelPort(address=0xC020)
 

        print(f"Experiment created for subject {self.subj}")

    def initiate_eeg(self):
        """Send TTL to mark in EEG before start of task."""
        self.logger.info("Start EEG recording and press E after that...")
        keys = event.waitKeys(keyList=["e"], timeStamped=self.clock, clearEvents=True)
        if self.parallel:
            self.port.setData(1)  # Beginning of EEG = 1 (S 01)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)
        eeg_time = keys[0][1]
        self.logger.log(
            level=SCANNER,
            msg=f"[[{eeg_time}]] EEG beginning mark sent at {eeg_time:.6f} s",
        )
        self._events.append((eeg_time, "EEG_START", -1))

    def _jittering(self):
        """Return a random jittering time."""
        # Calculate total extra time to distribute
        total_seconds = self.duration * 60
        base_time = self.n_trials * self.interval
        extra_time = total_seconds - base_time

        if extra_time < 0:
            raise ValueError("Duration too short for number of trials and interval")

        # Generate jitters around zero
        jitter_range = self.jittering
        jitter_values = [
            random.uniform(-jitter_range, jitter_range) for _ in range(self.n_trials)
        ]

        # Adjust so they sum to extra_time
        current_sum = sum(jitter_values)
        adjustment = (extra_time - current_sum) / self.n_trials
        jitter_values = [j + adjustment for j in jitter_values]

        random.shuffle(jitter_values)
        self.jittering = jitter_values

        return self.jittering

    def configure_logger(self):
        """Logger that prints messages to the console."""
        logger = logging.getLogger("ES_logger")
        logger.setLevel(SCANNER)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fh = logging.FileHandler(self.logfile)
        fh.setLevel(SCANNER)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

        logger.info(f"Logger configured for subject {self.subj}")
        logger.info(f"Log file: {self.logfile.as_posix()}")
        logger.info(f"Event file: {self.eventfile.as_posix()}")
        logger.info(f"Exp file: {self.expfile.as_posix()}")

        return logger

    def draw_cross(self):
        """Fixation cross stimulus."""

        fixation_cross = visual.GratingStim(

        )
        return fixation_cross

    def play_probe(self, trial_num):
        """Visual and auditory probe"""

        visual_probe = visual.TextStim(
            self.win, text="!", color="black", height=0.4, bold=True, pos=(0, 0.10)
        )

        # auditory_probe = sound.backend_ptb.SoundPTB(value=1000, secs=2, volume=0.5)
        visual_probe.draw()
        flip_time = self.win.flip()
        # auditory_probe.play(when=flip_time)


        if self.parallel:
            self.port.setData(0x2)  # Probe Time = 2 (S 02)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)


        probe_time = float(self.clock.getTime())
        self._events.append((probe_time, "PROBE_START", trial_num))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{probe_time}]] PROBE {trial_num + 1} played at {probe_time}s",
        )

        return probe_time

    def clear_key_buffer(self):
        # Clear buffer from response keys
        all_keys = event.getKeys(timeStamped=self.clock)

        # Log all scanner volumes
        for key, timestamp in all_keys:
            if key == "t":
                self._events.append((timestamp, "SCANNER", self._volume_count))
                self._volume_count += 1
                self.logger.log(
                    level=SCANNER,
                    msg=f"[[{timestamp}]] Volume {self._volume_count} received at {timestamp:.6f} s",
                )

                console_print_interval = 50  # volumes
                if self._volume_count % console_print_interval == 0:
                    self.logger.log(
                        level=SCANNER_SUMMARY,
                        msg=f"[[{timestamp}]] Volume {self._volume_count} received at {timestamp:.6f} s",
                    )

    def get_response(self, trial_num):
        """Display response options and collect participant response."""
        self.win.flip()

        prompt_txt = ""
        for i, state in enumerate(self.states, 1):
            prompt_txt += f"{state}\n\n"
        prompt = visual.TextStim(
            self.win,
            text=prompt_txt,
            color="black",
            height=0.09,
            wrapWidth=1,
            pos=(0, 0.005),
        )
        image = visual.ImageStim(
            win=self.win,
            image=picture_dir / "joystick_mri.png",
            pos=(0.60, 0.10),
            size=(0.50, 0.50),
        )
        image.draw()
        prompt.draw()
        self.clear_key_buffer()

        self.win.flip()
        prompt_onset = float(self.clock.getTime())
        self._events.append((prompt_onset, "PROMPT_START", trial_num))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{prompt_onset}]] PROMPT {trial_num + 1} displayed at {prompt_onset} s",
        )
        response = None
        keys = event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)

        response, response_time = keys[0]
        rt = response_time - prompt_onset

        if self.parallel:
            self.port.setData(0x4)  # Response Time = 3 (S 4)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)

        if response == "b":
            state_name = "Thought"
        elif response == "y":
            state_name = "Blank"
        elif response == "g":
            state_name = "Sleep"

        self._events.append((response_time, "RESPONSE", trial_num, response))

        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{response_time}]] RESPONSE {trial_num + 1}: '{state_name}' (key {response}), reaction time: {rt}s, raw time: {response_time}s",
        )
        feedback_text = f"You chose: {state_name}"
        feedback = visual.TextStim(
            self.win, text=feedback_text, color="black", height=0.09, pos=(0, 0.10)
        )
        feedback.draw()
        self.win.flip()
        core.wait(1)

        return prompt_onset, rt, state_name

    def get_arousal_rating(self, trial_num):
        """Display arousal rating scale with continuous slider."""

        # Question on the right side
        question = visual.TextStim(
            self.win,
            text="How awake do you feel right now?",
            color="black",
            height=0.05,
            pos=(0.00, 0.38),
        )

        # Labels at top and bottom of slider
        top_label = visual.TextStim(
            self.win, text="Very alert", color="black", height=0.05, pos=(0.0, 0.30)
        )

        bottom_label = visual.TextStim(
            self.win, text="Very sleepy", color="black", height=0.05, pos=(0.00, -0.10)
        )

        # Vertical slider line - positioned higher and to the left
        slider_line = visual.Line(
            self.win, start=(0, -0.05), end=(0, 0.25), lineColor="black", lineWidth=3
        )

        slider_marker = visual.Circle(
            self.win, radius=0.015, fillColor="black", lineColor="black", pos=(0, -0.05)
        )
        image = visual.ImageStim(
            win=self.win,
            image=picture_dir / "joystick_mri_arousal.png",
            pos=(0.60, 0.10),
            size=(0.50, 0.50),
        )
        self.clear_key_buffer()

        # Initialize rating and timing
        rating = 50
        confirmed = False
        prompt_onset = None

        while not confirmed:
            # Update slider position - MUST match slider_line range
            y_pos = -0.05 + (rating / 100) * 0.3  # From 0 to 0.3
            slider_marker.pos = (0, y_pos)

            # Draw everything
            image.draw()

            question.draw()
            top_label.draw()
            bottom_label.draw()
            slider_line.draw()
            slider_marker.draw()
            # Show current value next to slider
            value_text = visual.TextStim(
                self.win,
                text=f"{rating}%",
                color="black",
                height=0.04,
                pos=(0.08, y_pos),
            )
            value_text.draw()

            self.win.flip()
            if prompt_onset is None:
                prompt_onset = float(self.clock.getTime())
                self.logger.log(
                    level=EXPERIMENT,
                    msg=f"[[{prompt_onset}]] Arousal Prompt {trial_num + 1} displayed at {prompt_onset} s",
                )

            keys = event.waitKeys(keyList=["b", "y", "g"], timeStamped=self.clock)

            for key, timestamp in keys:
                if key == "g":
                    rating = max(0, rating - 5)
                    self.logger.log(
                        level=EXPERIMENT,
                        msg=f"Arousal rating {trial_num + 1}: rating decreased to {rating}",
                    )
                elif key == "b":
                    rating = min(100, rating + 5)
                    self.logger.log(
                        level=EXPERIMENT,
                        msg=f"Arousal rating {trial_num + 1}: rating increased to {rating}",
                    )
                elif key == "y":
                    confirmed = True
                    rating_time = timestamp
                    rt = rating_time - prompt_onset

        if self.parallel:
            self.port.setData(0x8)  # Response Time Arousal = 4 (S 8)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)

        self._events.append((rating_time, "RATING", rating, rt))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{rating_time}]] AROUSAL {trial_num + 1} final rating {rating}, reaction time: {rt} s, raw time: {rating_time}s",
        )

        self.win.flip()
        core.wait(0.1)
        return rating, rt

    def wait_scanner_trigger(self, n_triggers=5):
        """Wait for scanner trigger (e.g., '5' key press)."""
        self.logger.info(f"Waiting for {n_triggers} scanner triggers ...")
        trigger_times = []

        self.logger.info(f"Press 't' {n_triggers} times...")
        for i in range(n_triggers):
            self.logger.info(f"Waiting for trigger {i + 1}/{n_triggers}...")
            keys = event.waitKeys(
                keyList=["t", "escape"], timeStamped=self.clock, clearEvents=True
            )
            if any(x[0] == "escape" for x in keys):
                self.win.close()
                core.quit()
            trigger_time = keys[0][1]
            trigger_times.append(trigger_time)
            self._events.append((trigger_time, "SCANNER", self._volume_count))

            self.logger.log(
                level=SCANNER,
                msg=f"[[{trigger_time}]] VOLUME {self._volume_count} received at {trigger_time:.6f} s",
            )
            self.logger.log(
                level=SCANNER_SUMMARY,
                msg=f"[[{trigger_time}]] VOLUME {self._volume_count} received at {trigger_time:.6f} s",
            )
            self._volume_count += 1

        self.experiment_start_time = trigger_times[0]
        self.logger.info(f"Experiment T0: {self.experiment_start_time:.6f}")

    def run_trial(self, trial_num):
        """Run a single trial - handles all trial-specific operations."""
        self.logger.info(f"STARTING TRIAL {trial_num + 1} ...")

        # Show fixation cross for jittered interval
        self.cross.draw()
        self.win.flip()
        if self.parallel:
            self.port.setData(0x10)  # Response Time Arousal = 5 (S 16)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)

        trial_start = float(self.clock.getTime())
        self._events.append((trial_start, "TRIAL_START", trial_num))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{trial_start}]] Trial {trial_num + 1} started at {trial_start}",
        )

        trial_duration = self.jittering[trial_num] + self.interval
        self.logger.log(
            level=EXPERIMENT,
            msg=f"REST {trial_num + 1} duration of {trial_duration:.3f}s",
        )
        core.wait(trial_duration)

        # Play the probe
        probe_time = self.play_probe(trial_num)

        # Get response
        prompt_onset, rt, state = self.get_response(trial_num)

        # Get arousal rating only for Mind Blanking
        arousal, arousal_rt = None, None
        if state == "Blank":
            self.logger.log(
                level=EXPERIMENT,
                msg="Selected Mind Blanking, getting arousal rating...",
            )
            arousal, arousal_rt = self.get_arousal_rating(trial_num)

        # Return trial data
        return {
            "trial_num": trial_num + 1,
            "probe_onset": probe_time,
            "prompt_onset": prompt_onset,
            "rest_duration": trial_duration,
            "state": state,
            "response_rt": rt,
            "arousal": arousal,
            "arousal_rt": arousal_rt,
        }

    def run_experiment(self):
        """Main experiment loop - coordinates the overall flow."""
        # Start EEG recording
        self.initiate_eeg()
        # Wait for scanner triggers
        self.wait_scanner_trigger(n_triggers=5)
        all_trials_data = []
        self._jittering()
        # Run all trials
        for trial_num in range(self.n_trials):
            trial_data = self.run_trial(trial_num)

            # Log trial completion with timing info
            trial_end = float(self.clock.getTime())
            self.logger.log(
                level=EXPERIMENT,
                msg=f"Trial {trial_num + 1} completed at {trial_end:.3f}s after sync",
            )
            self._events.append((trial_end, "TRIAL_END", trial_num))
            all_trials_data.append(trial_data)

        self.clear_key_buffer()

        if self.parallel:
            self.port.setData(0x20)  # End of EEG = 6 (S 32)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)

        finished = False
        self.logger.info(f"END OF TASK. Volume count: {self._volume_count}")

        while not finished:
            visual_probe = visual.TextStim(
                self.win,
                text=f"Experiment Done. # Volumes = {self._volume_count}. Stop the fMRI Sequence and press f to save the results.",
                color="black",
                height=0.05,
                bold=True,
            )

            visual_probe.draw()
            self.win.flip()

            keys = event.getKeys(["f", "t"], timeStamped=self.clock)
            for key, timestamp in keys:
                if key == "f":
                    finished = True
                elif key == "t":
                    self._events.append((timestamp, "SCANNER", self._volume_count))
                    self._volume_count += 1
            core.wait(0.1)

        self.clear_key_buffer()
        self.logger.log(
            level=SCANNER_SUMMARY,
            msg=f"Volume count: {self._volume_count} (total volumes)",
        )
        self.logger.info("Experiment completed successfully!")
        self.win.close()

        # Save all_trials_data
        with open(self.expfile, "w") as f_exp:
            yaml.dump(all_trials_data, f_exp)

        # Save events
        sorted_events = sorted(self._events, key=lambda x: x[0])
        list_events = [list(x) for x in sorted_events]
        with open(self.eventfile, "w") as f_exp:
            yaml.dump(list_events, f_exp)

        # Copy exp file to subjects folder
        shutil.copy(Path(__file__), self.exp_dir)


################################
################################
# Example usage:
################################
################################

# if __name__ == "__main__":
#     params = {
#         "subj": "001",
#         "n_trials": 50,
#         "interval": 45,
#         "jittering": 15,
#         "duration": 40,  # total duration in minutes,
#         "states": ["Thought", "Blank", "Asleep"],
#         "parallel": True,
#         "response_buttons": ["b", "y", "g"],
#     }
#     experiment = experience_sampling(params)
#     experiment.run_experiment()
