from psychopy import parallel
from psychopy.core import wait
port = parallel.ParallelPort(address=0x0378)
port.setData(0)  # start at 0

for value in [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]:
    port.setData(value)  # set the value to 0x0F (15 in decimal)
    wait(1)          # wait 1s
    port.setData(0)
    wait(1)