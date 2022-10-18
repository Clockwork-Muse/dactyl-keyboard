FROM docker.io/ubuntu:22.04

# Avoid warnings by switching to noninteractive
ARG DEBIAN_FRONTEND=noninteractive

RUN \
    #
    # Create directories in the skel directory
    mkdir -p /etc/skel/.vscode-server/extensions \
    && mkdir -p /etc/skel/.vscode-server-insiders/extensions

RUN \
    #
    # Update and install packages
    apt-get update --yes --fix-missing \
    && apt-get install --yes --quiet --no-install-recommends \
        libgl1-mesa-glx \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
