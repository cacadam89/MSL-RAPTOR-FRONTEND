FROM nvcr.io/nvidia/l4t-base:r32.2.1

ENV TZ=America/Los_Angeles
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y python3-pip libatlas-base-dev git libjpeg-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/* 

RUN wget https://nvidia.box.com/shared/static/phqe92v26cbhqjohwtvxorrwnmrnfx1o.whl -O torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    pip3 install torch-1.3.0-cp36-cp36m-linux_aarch64.whl && \
    git clone --branch v0.5.0 https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install

# Torch to TensorRT package
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python3 setup.py install


# SiamMask
RUN git clone -b tensorrt https://github.com/bramtoula/SiamMask.git && \
    cd SiamMask && \
    apt-get update && \
    apt-get install -y python3-h5py python3-scipy python3-llvmlite python3-pandas python3-numba && \
    pip3 install Cython==0.29.4 colorama==0.3.9 requests==2.21.0 fire==0.1.3 tqdm==4.29.1 tensorboardX==1.6 && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth && \
    rm -rf /var/lib/apt/lists/* 