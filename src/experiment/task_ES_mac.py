# from psychopy import prefs
# prefs.hardware['audioDevice'] = 'SPDIF-Out (SB Recon3D PCIe)'
# import psychtoolbox.audio
import yaml
import logging
import random
from pathlib import Path
import shutil
import numpy as np

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

RANDOM_JITTERS = np.array([
    0.52652746,
    0.15729363,
    0.75443592,
    0.060392,
    0.35669565,
    0.6807377,
    0.73926642,
    0.15586217,
    0.47128575,
    0.71444614,
    0.4809827,
    0.58392067,
    0.17503814,
    0.2759395,
    0.8467543,
    0.63257164,
    0.56914393,
    0.85239554,
    0.61479426,
    0.6232063,
    0.69778271,
    0.31498446,
    0.27391724,
    0.41028833,
    0.86310978,
    0.13499124,
    0.89001996,
    0.31498447,
    0.67315201,
    0.62856914,
    0.76812966,
    0.09887945,
    0.43616935,
    0.49564682,
    0.94048908,
    0.50164306,
    0.83092926,
    0.75208337,
    0.63359351,
    0.08121215,
    0.51719137,
    0.40122761,
    0.4463277,
    0.26206885,
    0.52061206,
    0.24690058,
    0.50681413,
    0.85429209,
    0.32270196,
    0.05197819,
    0.99644992,
    0.73888312,
    0.08686468,
    0.6325577,
    0.36590586,
    0.70881948,
    0.97023796,
    0.69525613,
    0.1529349,
    0.48179428,
    0.91902002,
    0.37082624,
    0.83830293,
    0.93963037,
    0.93150584,
    0.6887494,
    0.49496108,
    0.67194435,
    0.05938193,
    0.72451566,
    0.29710184,
    0.08341319,
    0.65451143,
    0.11456868,
    0.74371139,
    0.41923367,
    0.1164916,
    0.66932978,
    0.00931658,
    0.39217487,
    0.41103187,
    0.76287093,
    0.11590666,
    0.93085323,
    0.92980592,
    0.67747251,
    0.96370689,
    0.72976788,
    0.67509138,
    0.67489054,
    0.60635407,
    0.52165654,
    0.4089209,
    0.99235989,
    0.83686703,
    0.86539419,
    0.21215573,
    0.00713328,
    0.95010646,
    0.21432487,
])


class experience_sampling:
    """A class to run a MB experience sampling task."""

    def __init__(self, params):
        # Task parameters
        self.subj = params["subj"]
        self.n_trials = params["n_trials"]
        self.interval = params["interval"]
        self.states = params["states"]
        self.parallel = params["parallel"]
        self.response_buttons = params["response_buttons"]
        self.exp_dir = Path(__file__).parent / f"sub-{self.subj}" / "task-ES"
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
        jitter_range = self.jittering

        jitter_values = (RANDOM_JITTERS[: self.n_trials]* 2-1) * jitter_range
        np.random.shuffle(jitter_values)

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

    def instructions(self):
        """Display instructions to the participant."""
        instructions_ES_text = (
            "Before we start, let's remember the instructions.\n\n"
            "During the task: let your mind wander freely, keep your eyes on the cross (+), and stay still.\n\n"
            "Press any key to continue (Page 1 of 6)."
        )
        instructions_ES2_text = (
            "When '!' appears, indicate your mental state just before it:\n\n"
        )
        instructions_ES2options_text = (
            "1. Thought (Index): Thinking about something\n"
            "2. Blank (Middle): Mind was blank, no thought you can spot\n"
            "3. Sleep (Ring): Feeling drowsy or asleep\n"
            "4. Sensation (Little): Noticing the environment or body sensations\n\n"
        )
        instructions_ES2continue_text = "Press any key to continue (Page 2 of 6)."
        instructions_image_text = "This is how the screen will look like:"
        instructions_imageEScontinue_text = "Press any key to continue (Page 3 of 6)."
        instructions_arousaltext = (
            "Next, you will rate how awake you feel from 0 (very sleepy) to 100 (very alert).\n\n"
            "Index/Ring: Adjust slider | Middle: Confirm\n\n"
            "Press any key to continue (Page 4 of 6)."
        )

        instructions_imagearousalcontinue_text = (
            "Press any key to continue (Page 5 of 6)."
        )

        instructions_readytostart_text = (
            "Remember: just let your mind wander freely :)\n\n"
            "When you are ready, press any key to begin (Page 6 of 6)."
        )
        instructions_startsoon_text = "The experiment will start in a few seconds :)"
        image_ES = visual.ImageStim(
            win=self.win,
            image=picture_dir / "instructions_ES.png",
            pos=(0, 0),
            size=(1.2, 0.65),
        )
        image_arousal = visual.ImageStim(
            win=self.win,
            image=picture_dir / "instructions_arousal.png",
            pos=(0, 0),
            size=(1.3, 0.65),
        )
        # Page 1
        instr1 = visual.TextStim(
            self.win,
            text=instructions_ES_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.15),
        )
        instr1.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 2
        instr1b = visual.TextStim(
            self.win,
            text=instructions_ES2_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.3),
        )
        instr1b_options = visual.TextStim(
            self.win,
            text=instructions_ES2options_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.10),
            alignText="left",
        )
        instr1b_continue = visual.TextStim(
            self.win,
            text=instructions_ES2continue_text,
            color="black",
            height=0.05,
            wrapWidth=1.0,
            pos=(0, -0.02),
        )
        instr1b.draw()
        instr1b_options.draw()
        instr1b_continue.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 3
        instr2 = visual.TextStim(
            self.win,
            text=instructions_image_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.40),
        )
        instr2_next = visual.TextStim(
            self.win,
            text=instructions_imageEScontinue_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, -0.36),
        )
        instr2.draw()
        image_ES.draw()
        instr2_next.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 4
        instr3 = visual.TextStim(
            self.win,
            text=instructions_arousaltext,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.15),
        )
        instr3.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 5
        instr4 = visual.TextStim(
            self.win,
            text=instructions_image_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.40),
        )
        instr4_next = visual.TextStim(
            self.win,
            text=instructions_imagearousalcontinue_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, -0.36),
        )
        instr4.draw()
        instr4_next.draw()
        image_arousal.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 6
        instr5 = visual.TextStim(
            self.win,
            text=instructions_readytostart_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.15),
        )
        instr5.draw()
        self.win.flip()
        event.waitKeys(keyList=self.response_buttons, timeStamped=self.clock)
        # Page 7
        instr6 = visual.TextStim(
            self.win,
            text=instructions_startsoon_text,
            color="black",
            height=0.05,
            wrapWidth=1.7,
            pos=(0, 0.15),
        )
        instr6.draw()
        self.win.flip()

    def draw_cross(self):
        """Fixation cross stimulus."""

        fixation_cross = visual.GratingStim(
            win=self.win, size=0.1, pos=[0, 0.10], sf=0, color="black", mask="cross"
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
            height=0.08,
            pos=(0, 0.008),
        )
        image = visual.ImageStim(
            win=self.win,
            image=picture_dir / "joystick_mri.png",
            pos=(0.60, 0.13),
            size=(0.65, 0.65),
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
        elif response == "r":
            state_name = "Sensation"

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
            height=0.063,
            pos=(0.00, 0.40),
        )

        # Labels at top and bottom of slider
        top_label = visual.TextStim(
            self.win, text="Very alert", color="black", height=0.06, pos=(0.0, 0.30)
        )

        bottom_label = visual.TextStim(
            self.win, text="Very sleepy", color="black", height=0.06, pos=(0.00, -0.10)
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
            pos=(0.60, 0.05),
            size=(0.65, 0.65),
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
                height=0.05,
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

    def _save_events(self):
        """Save events to YAML file."""
        try:
            sorted_events = sorted(self._events, key=lambda x: x[0])
            list_events = [list(x) for x in sorted_events]
            with open(self.eventfile, "w") as f:
                yaml.dump(list_events, f)
        except Exception as e:
            self.logger.error(f"Failed to save events: {e}")

    def _save_trial_data(self, all_trials_data):
        """Save trial data to YAML file."""
        try:
            with open(self.expfile, "w") as f:
                yaml.dump(all_trials_data, f)
        except Exception as e:
            self.logger.error(f"Failed to save trial data: {e}")

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
        self.logger.log(
            level=EXPERIMENT,
            msg="Getting arousal rating...",
        )
        arousal, arousal_rt = self.get_arousal_rating(trial_num)

        # Return trial data
        return {
            "trial_num": trial_num + 1,
            "probe_onset": probe_time,
            "prompt_onset": prompt_onset,
            "rest_duration": float(trial_duration),
            "state": state,
            "response_rt": rt,
            "arousal": arousal,
            "arousal_rt": arousal_rt,
        }

    def run_experiment(self):
        """Main experiment loop - coordinates the overall flow."""
        # Instructions
        self.instructions()
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
            self._save_events()
            self._save_trial_data(all_trials_data)

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

        # Final save (redundant but safe)
        self._save_trial_data(all_trials_data)
        self._save_events()

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
