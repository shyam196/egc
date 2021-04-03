FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Install some basic utilities
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    vim \
    nano \
    libglib2.0-0 \
    rsync\
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
    && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -Lso ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.11.0-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.7 environment
RUN /home/user/miniconda/bin/conda create -y --name py37 python=3.7 \
    && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda install -y pytorch==1.11 torchvision torchaudio cudatoolkit=11.3 -c pytorch \
    && conda clean -ya

# I have no idea why it's not possible to install PyG with Conda -- but seemingly
# conda's dependency resolver complains. Anyway, this is a workaround.
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    torch-geometric==2.0 -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

RUN mkdir -p /app/third_party/exptune
ADD third_party/exptune /app/third_party/exptune
RUN pip install /app/third_party/exptune

RUN mkdir -p /app/main/experiments
ADD requirements.txt /app/main/
RUN pip install -r /app/main/requirements.txt
ADD run_pretrained.sh /app/main
ADD train_ablation.sh /app/main
ADD train_main_table.sh /app/main

ADD main.py /app/main/
ADD experiments /app/main/experiments

ENV DATA_LOC=/app/datasets
RUN mkdir -p /app/datasets
ENV PYTHONPATH="/app/main:${PYTHONPATH}"

WORKDIR /app/main
