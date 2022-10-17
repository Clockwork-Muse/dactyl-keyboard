#!/usr/bin/env bash

# Create python environment
if [ ! -d ${WORKSPACE}/.venv ]; then
    python -m venv ${WORKSPACE}/.venv
fi
source ${WORKSPACE}/.venv/bin/activate
if [ -f ${WORKSPACE}/.pyproject.toml ]; then
    pip install -e ${WORKSPACE}
fi
