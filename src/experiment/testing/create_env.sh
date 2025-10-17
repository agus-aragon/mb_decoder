#!/bin/bash

# This script creates a Python virtual environment for the project

cmd="uv"
if [ ! "$(command -v $cmd)" ]; then
    echo "command \"$cmd\" does not exist on system"
    exit -1
fi

$cmd venv
$cmd pip install -r requirements.txt