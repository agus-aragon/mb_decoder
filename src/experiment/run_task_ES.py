###
# Run Task "Experience Sampling"
# INPUT:
#
#
#
# DESCRIPTION:
# This script runs the Mind Blanking Experience Sampling task in the scanner.
#
# OUTPUT:
# - A log file with the task results.
###

#!/usr/bin/env python3

#%% Initialization
import os
import sys
import time
import numpy as np
import logging
from psychopy import visual, core, event, logging