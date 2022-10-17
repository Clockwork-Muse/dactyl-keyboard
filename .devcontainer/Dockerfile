FROM docker.io/ubuntu:22.04

# Avoid warnings by switching to noninteractive
ARG DEBIAN_FRONTEND=noninteractive

RUN \
    #
    # Create directories in the skel directory
    mkdir -p /etc/skel/.vscode-server/extensions \
    && mkdir -p /etc/skel/.vscode-server-insiders/extensions
