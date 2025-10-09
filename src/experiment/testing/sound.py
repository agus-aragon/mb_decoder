from psychopy import prefs
prefs.hardware['audioLib'] = ['PTB', 'sounddevice', 'pyo']
prefs.hardware['audioLatencyMode'] = 3

import sounddevice as sd
from psychopy import visual, sound
from psychopy.sound import backend_ptb
import time

win = visual.Window(size=(1270, 720), units="height", color="white")

# Get all output devices
devices = sd.query_devices()

for i, device in enumerate(devices):

    print(f"\nTesting device {i}: {device['name']}")
    
    # Display which device is being tested
    text = visual.TextStim(
        win, 
        text=f"{i}: {device['name']}", 
        color="black", 
        height=0.04, 
        bold=True, 
        pos=(0, 0.0),
        wrapWidth=0.9
    )
    text.draw()
    win.flip()
    
    sd.default.device = i
    auditory_probe = sound.backend_ptb.SoundPTB(
        value='C',
        secs=2,
        volume=1
    )
    auditory_probe.play()
    time.sleep(2.5)  

win.close()