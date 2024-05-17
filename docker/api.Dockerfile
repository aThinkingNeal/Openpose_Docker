FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04 as builder

LABEL maintainer="Tiancheng Huang <athinkingneal@gmail.com>" \
    lastupdate="2024-05-14"

# Set the timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update package lists and install wheels
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
        git wget curl zip unzip bzip2 vim inetutils-ping sudo net-tools iproute2 \
        build-essential \
        libgl1 libglib2.0-0 libssl-dev libcurl4-openssl-dev libgl1-mesa-glx \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "--login", "-c"]

RUN cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" && \
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    /opt/conda/bin/conda create -n venv python=3.11

ENV PATH $PATH:/opt/conda/envs/venv/bin
    
RUN conda init bash &&\
    echo "conda activate venv" >> ~/.bashrc &&\
    conda activate venv &&\
    pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git \
    opencv-contrib-python controlnet_aux matplotlib mediapipe pandas openpyxl openai python-dotenv flask

# Set the working directory
WORKDIR /workspace

ARG GITHUB_TOKEN

# Clone the private repository
RUN git clone https://${GITHUB_TOKEN}@github.com/aThinkingNeal/Openpose_Docker.git

# Install other dependencies and download model weights
RUN python -c "from controlnet_aux import OpenposeDetector; OpenposeDetector.from_pretrained('lllyasviel/ControlNet')"

# Use multi-stage build to reduce the image size and hide the ARG
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

ENV TIME_ZONE="Asia/Shanghai" \
    TZ="Asia/Shanghai" \
    CUDA_TOOLKIT_ROOT_DIR="/usr/local/cuda" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    PYTHONPATH="/workspace" \
    EXEC_PROVIDER="CUDAExecutionProvider" 

# os packages and timezone
RUN apt-get -y update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libssl-dev libcurl4-openssl-dev curl \
        tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy Conda and environment from builder
COPY --from=builder /opt/conda /opt/conda

# Copy workspace from builder
COPY --from=builder /workspace /workspace

# Set the environment variables for Conda
ENV PATH /opt/conda/envs/venv/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=venv

# Activate the conda environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate venv" >> ~/.bashrc

# Expose the port that Flask will run on
EXPOSE 5000

# Set the working directory
WORKDIR /workspace

# Run the Flask workspacelication
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate venv && python Openpose_Docker/predict.py"]
