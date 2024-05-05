FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

LABEL maintainer="Tiancheng Huang <athinkingneal@gmail.com>" \
    lastupdate="2024-04-24"

# Set the timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install some basic utilities
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        git wget curl zip unzip bzip2 vim inetutils-ping sudo net-tools iproute2 \
        build-essential \
        libgl1 libglib2.0-0 libssl-dev libcurl4-openssl-dev \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python 3.11
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv ca-certificates \
    python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip

# create the workspace
RUN mkdir -p /workspace
WORKDIR /workspace

RUN python3.11 -m venv /workspace/venv
ENV VIRTUAL_ENV=/workspace/venv \
    PATH="/workspace/venv/bin:$PATH"

RUN pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git \
    opencv-contrib-python controlnet_aux matplotlib mediapipe pandas openpyxl openai python-dotenv


