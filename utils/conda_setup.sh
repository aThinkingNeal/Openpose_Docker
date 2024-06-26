
 cd / && wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    /bin/bash /miniconda.sh -b -p /opt/conda &&\
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh &&\
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
    /bin/bash -c "source ~/.bashrc" && \
    /opt/conda/bin/conda update -n base -c defaults conda -y &&\
    /opt/conda/bin/conda create -n venv python=3.11


PATH $PATH:/opt/conda/envs/venv/bin
    
conda init bash &&\
    echo "conda activate venv" >> ~/.bashrc &&\
    conda activate venv &&\
    pip install -q diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git \
    opencv-contrib-python controlnet_aux matplotlib mediapipe pandas openpyxl openai python-dotenv flask
