FROM nvcr.io/nvidia/tensorrt:19.11-py3

RUN git clone -b tensorrt https://github.com/bramtoula/SiamMask.git && \
    cd SiamMask && \
    pip3 install -r requirements.txt && \
    /bin/bash make.sh && \
    cd experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth  && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth


# RUN git clone -b tensorrt https://github.com/bramtoula/SiamMask.git
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && \
    cd torch2trt && \
    python setup.py install

RUN apt-get update && \
    apt-get install -y libsm6 libxext6 libxrender-dev
