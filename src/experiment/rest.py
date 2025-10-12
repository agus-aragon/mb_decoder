import yaml
import logging
from pathlib import Path
import shutil

# from psychtoolbox import audio
# import psychtoolbox as ptb
from psychopy import visual, core, event, parallel


EEG = 25
SCANNER = 26
logging.addLevelName(SCANNER, "SCANNER")
logging.addLevelName(EEG, "EEG")


class resting_state:
    """A class to run a MB experience sampling task."""

    def __init__(self, params):
        # Task parameters
        self.subj = params["subj"]
        self.parallel = params["parallel"]
        self.duration = params["duration"]
        self.exp_dir = Path(__file__).parent / f"sub-{self.subj}" / "rest"
        if self.exp_dir.exists():            
            raise ValueError("An experiment dir for this subject already exists")
        self.exp_dir.mkdir(parents=True)
        self.logfile = self.exp_dir / f"sub_{self.subj}_rest_log.log"
        self.eventfile = self.exp_dir / f"sub_{self.subj}_rest_ev.yaml"
        self.expfile = self.exp_dir / f"sub_{self.subj}_rest_exp.yaml"

        self._volume_count = 0
        self._events = []

        self.win = visual.Window(size=(1010, 520), units="height", color="white")
        self.clock = core.Clock()
        self.logger = self.configure_logger()
        self.cross = self.draw_cross()

        # parallel settings
        if self.parallel:
            self.port = parallel.ParallelPort(address=0xC020)
            self.port.setData(0)
        print(f"Rest created for subject {self.subj}")

    def initiate_eeg(self):
        """Send TTL to mark in EEG before start of task."""
        self.logger.info("Start EEG recording and press E after that...")
        print("Press 'e' to start the EEG ...")
        if self.parallel:
            self.port.setData(0x1)  # Beginning of EEG = 1 (S 01)
            core.wait(0.1)
            self.port.setData(0)
            core.wait(0.1)
        keys = event.waitKeys(keyList=["e"], timeStamped=self.clock, clearEvents=True)
        eeg_time_start = keys[0][1]
        self.logger.log(
            level=EEG,
            msg=f"[[{eeg_time_start}]] EEG beginning mark sent at {eeg_time_start:.6f} s",
        )
        print("Start fMRI sequence ...")
        self._events.append((eeg_time_start, "EEG_START", -1))

    def configure_logger(self):
        """Logger that prints messages to the console."""
        logger = logging.getLogger("ES_logger")
        logger.setLevel(EEG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fh = logging.FileHandler(self.logfile)
        fh.setLevel(EEG)

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
            win=self.win, size=0.1, pos=[0, 0.10], sf=0, color="black", mask="cross"
        )
        fixation_cross.draw()
        self.win.flip()
        return fixation_cross

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
            self._volume_count += 1


        self.experiment_start_time = trigger_times[0]
        self.logger.info(f"T0: {self.experiment_start_time:.6f}")

    def run_rest(self):
        """Main rest loop - coordinates the overall flow."""
        # Start EEG recording
        self.initiate_eeg()
        # Wait for scanner triggers
        self.wait_scanner_trigger(n_triggers=5)
        self.draw_cross()
        self.clear_key_buffer()
        print('Press "f" to finish the rest')
        finished = False
        while not finished:
            keys = event.getKeys(["f", "t"], timeStamped=self.clock)
            for key, timestamp in keys:
                if key == "f":
                    finished = True
                    if self.parallel:
                        self.port.setData(0x20)  # End of EEG = 6 (S 32)
                        core.wait(0.1)
                        self.port.setData(0)
                        core.wait(0.1)
                        eeg_time_end = keys[0][1]
                        self.logger.log(
                            level=EEG,
                            msg=f"[[{eeg_time_end}]] EEG end mark sent at {eeg_time_end:.6f} s",
                        )
                        self._events.append((eeg_time_end, "EEG_END", -1))

                    self.logger.info(f"END OF REST. Volume count: {self._volume_count}")
                elif key == "t":
                    self._events.append((timestamp, "SCANNER", self._volume_count))
                    self._volume_count += 1
            core.wait(0.1)

        self.clear_key_buffer()
        self.logger.log(
            level=SCANNER,
            msg=f"Volume count: {self._volume_count} (total volumes)",
        )
        self.logger.info("Rest completed successfully!")
        self.win.close()

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
#         "duration": 0.20,  # total duration in minutes,
#         "parallel": False,
#     }
#     resting = resting_state(params)
#     resting.run_rest()
