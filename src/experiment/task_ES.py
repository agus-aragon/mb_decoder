import yaml
import logging
import random
from pathlib import Path
import shutil
from psychopy import visual, core, event, sound, parallel

SCANNER = 15
EXPERIMENT = 25
SCANNER_SUMMARY = 26
logging.addLevelName(SCANNER, "SCANNER")
logging.addLevelName(SCANNER_SUMMARY, "SCANNER_SUMMARY")
logging.addLevelName(EXPERIMENT, "EXPERIMENT")


class experience_sampling:
    """A class to run a MB experience sampling task."""

    def __init__(self, params):
        # Task parameters
        self.subj = params["subj"]
        self.n_trials = params["n_trials"]
        self.interval = params["interval"]
        self.states = params["states"]
        self.parallel = params["parallel"]
        self.task_length = int((self.n_trials * self.interval) / 60)
        self.response_buttons = params["response_buttons"]
        self.exp_dir = Path(__file__).parent / f"sub-{self.subj}"
        if self.exp_dir.exists():
            raise ValueError("An experiment dir for this subject already exists")
        self.exp_dir.mkdir()
        self.logfile = self.exp_dir / f"sub_{self.subj}_task-ES_log.log"
        self.eventfile = self.exp_dir / f"sub_{self.subj}_task-ES_ev.yaml"
        self.expfile = self.exp_dir / f"sub_{self.subj}_task-ES_exp.yaml"
        self.jittering = params["jittering"]

        self._volume_count = 0
        self._events = []

        self.win = visual.Window(size=(1270, 720), units="height", color="white")
        self.clock = core.Clock()
        self.logger = self.configure_logger()
        self.cross = self.draw_cross()


        # parallel settings
        if self.parallel:
            self.port = parallel.ParallelPort(address=0x0378)
            self.port.setData(0)

        print(f"Experiment created for subject {self.subj}")

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
            win=self.win, size=0.1, pos=[0, 0], sf=0, color="black", mask="cross"
        )
        return fixation_cross

    def play_probe(self, trial_num):
        """Visual and auditory probe"""

        visual_probe = visual.TextStim(
            self.win, text="!", color="black", height=0.4, bold=True
        )

        auditory_probe = sound.Sound(value=1000,
                                     secs=1,
                                     volume=0.7)
        auditory_probe.play()
        visual_probe.draw()
        self.win.flip()
        probe_time = float(self.clock.getTime())
        self._events.append((probe_time, "PROBE_START", trial_num))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{probe_time}]] PROBE {trial_num + 1} played at {probe_time}s",
        )
        core.wait(1)
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
            height=0.1,
            wrapWidth=1.8,
            pos=(0, -0.1)
        )

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

        if response == "b":
            state_name = "Thought"
        elif response == "y":
            state_name = "Mind Blanking"
        elif response == "g":
            state_name = "Sleep"

        self._events.append((response_time, "RESPONSE", trial_num, response))

        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{response_time}]] RESPONSE {trial_num + 1}: '{state_name}' (key {response}), reaction time: {rt}s, raw time: {response_time}s",
        )
        feedback_text = f"You chose: {state_name}"
        feedback = visual.TextStim(
            self.win, text=feedback_text, color="black", height=0.1
        )
        feedback.draw()
        self.win.flip()
        core.wait(1)

        return prompt_onset, rt, state_name

    def get_arousal_rating(self, trial_num):
        """Display arousal rating scale with continuous slider."""

        # Create scale elements
        question = visual.TextStim(
            self.win,
            text="How awake do you feel right now?",
            color="black",
            height=0.07,
            pos=(0, 0.3),
        )

        left_label = visual.TextStim(
            self.win, text="Very sleepy", color="black", height=0.05, pos=(-0.4, 0.15)
        )

        right_label = visual.TextStim(
            self.win, text="Very alert", color="black", height=0.05, pos=(0.4, 0.15)
        )

        instruction = visual.TextStim(
            self.win,
            text="Use index and middle fingers to move, then press with the ring finger to confirm",
            color="black",
            height=0.04,
            pos=(0, -0.3),
        )

        # Create slider
        slider_line = visual.Line(
            self.win, start=(-0.4, 0), end=(0.4, 0), lineColor="black", lineWidth=2
        )

        slider_marker = visual.Circle(
            self.win, radius=0.02, fillColor="red", lineColor="red", pos=(0, 0)
        )

        self.clear_key_buffer()

        # if self.parallel:
        #     self.port.setData(0x40)
        #     core.wait(0.01)
        #     self.port.setData(0)

        # Initialize rating and timing
        rating = 50
        confirmed = False
        prompt_onset = None
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

            # Show current value
            value_text = visual.TextStim(
                self.win, text=f"{rating}%", color="black", height=0.05, pos=(0, -0.1)
            )
            value_text.draw()

            self.win.flip()
            if prompt_onset is None:
                prompt_onset = float(self.clock.getTime())
                self.logger.log(
                    level=EXPERIMENT,
                    msg=f"[[{prompt_onset}]] Arousal Prompt {trial_num + 1} displayed at {prompt_onset} s",
                )
            # Check for key presses
            keys = event.waitKeys(keyList=["1", "2", "3"], timeStamped=self.clock)

            for key, timestamp in keys:
                if key == "1":
                    rating = max(0, rating - 5)  # Decrease by 5%
                    self.logger.log(
                        level=EXPERIMENT,
                        msg=f"Arousal rating {trial_num + 1}: rating decreased to {rating}",
                    )
                elif key == "2":
                    rating = min(100, rating + 5)  # Increase by 5%
                    self.logger.log(
                        level=EXPERIMENT,
                        msg=f"Arousal rating {trial_num + 1}: rating increased to {rating}",
                    )
                elif key == "3":
                    confirmed = True
                    rating_time = timestamp
                    rt = rating_time - prompt_onset
        self._events.append((rating_time, "RATING", rating, rt))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{rating_time} AROUSAL {trial_num + 1} final rating {rating}, reaction time: {rt} s, raw time: {rating_time}s",
        )

        # if self.parallel:
        #     rating_code = 0x40 + int(rating)
        #     self.port.setData(rating_code)
        #     core.wait(0.001)
        #     self.port.setData(0)

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
        trial_start = float(self.clock.getTime())
        self._events.append((trial_start, "TRIAL_START", trial_num))
        self.logger.log(
            level=EXPERIMENT,
            msg=f"[[{trial_start}]] Trial {trial_num + 1} started at {trial_start}",
        )

        trial_duration = random.uniform(
            self.interval - self.jittering, self.interval + self.jittering
        )
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
        if state == "Mind Blanking":
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
        # Wait for scanner triggers
        self.wait_scanner_trigger(n_triggers=5)
        all_trials_data = []

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

if __name__ == "__main__":
    params = {
        "subj": "001",
        "n_trials": 3,
        "interval": 2,
        "jittering": 1,
        "states": ["Thought", "Mind Blanking", "Asleep"],
        "parallel": False,
        "response_buttons": ["b", "y", "g"],
    }
x
    experiment = experience_sampling(params)
    experiment.run_experiment()
