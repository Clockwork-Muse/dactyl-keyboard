#!/usr/bin/env bash

# Create python environment
if [ ! -d ${WORKSPACE}/.venv ]; then
    python -m venv ${WORKSPACE}/.venv
    source ${WORKSPACE}/.venv/bin/activate
    python -m pip install --upgrade pip
fi

source ${WORKSPACE}/.venv/bin/activate
pip install -e ${WORKSPACE}[cadquery,solid]
