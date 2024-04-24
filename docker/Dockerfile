FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

LABEL maintainer="Tiancheng Huang <athinkingneal@gmail.com>" \
    lastupdate="2024-04-24"

# Set the timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install some basic utilities
RUN apt-get update --fix-missing && \
    apt-get install -y libgtk2.0-dev && \
    apt-get install -y wget bzip2 ca-certificates curl git vim tmux g++ gcc build-essential cmake checkinstall gfortran libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libv4l-dev qt5-default libgtk2.0-dev libtbb-dev libatlas-base-dev libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev libgphoto2-dev libhdf5-dev doxygen libflann-dev libboost-all-dev proj-data libproj-dev libyaml-cpp-dev cmake-curses-gui libzmq3-dev freeglut3-dev

# install python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv ca-certificates

# create the workspace
RUN mkdir -p /workspace
WORKDIR /workspace

RUN python3.11 -m venv /workspace/venv
ENV VIRTUAL_ENV=/workspace/venv \
    PATH="/workspace/venv/bin:$PATH"

# get the requirements from the repo and install
COPY requirements.txt /workspace/requirements.txt 
RUN pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt


