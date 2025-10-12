from psychopy import visual, event, core

class preparation:
    """A class to display preparation messages between tasks."""
    
    def __init__(self):
        self.win = visual.Window(size=(1010, 520), units="height", color="white")
    
    def run_preparation(self):
        """Display preparation messages before starting the ES task."""
        # Show "Soon we start" message
        message1 = visual.TextStim(
            self.win,
            text="Soon we start with the task :)",
            color="black",
            height=0.06,
            wrapWidth=0.9
        )
        message1.draw()
        self.win.flip()
        
        print("Press 'c' to start the ES task ...")
        event.waitKeys(keyList=['c'])
        
        # Show "Starting" message
        message2 = visual.TextStim(
            self.win,
            text="Starting with the task!",
            color="black",
            height=0.06
        )
        message2.draw()
        self.win.flip()
        core.wait(2)
        
        self.win.close()

