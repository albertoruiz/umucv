# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.231.6/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version (use hirsuite or bionic on local arm64/Apple Silicon): hirsute, focal, bionic
ARG VARIANT="hirsute"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

COPY requirements.txt requirements.txt
# [Optional] Uncomment this section to install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python-is-python3 python3-pip

RUN pip install -r requirements.txt

RUN apt-get -y install ffmpeg libgstreamer1.0-dev libva-dev libsm6 libxext6 libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libswresample-dev libswscale-dev libva-drm2 libva-glx2 libva-wayland2 libva-x11-2 libgl1-mesa-dri vainfo

