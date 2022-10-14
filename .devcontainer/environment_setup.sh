#!/usr/bin/env bash

# Create conda environment
/opt/conda/condabin/conda init --user bash
if [ ! -d ${WORKSPACE}/.venv ]; then
    /opt/conda/condabin/conda env create --prefix ${WORKSPACE}/.venv -f .devcontainer/conda-environment.yml
fi

echo "Run \"conda activate \${WORKSPACE}/.venv\" to activate the environment."
