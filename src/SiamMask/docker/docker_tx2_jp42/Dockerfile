FROM ubuntu:18.04
ENV TZ=America/Los_Angeles
ENV DEBIAN_FRONTEND=noninteractive
# This assumes we are using Jetpack 4.2 and Tegra Docker (https://github.com/Technica-Corporation/Tegra-Docker)

# Basic stuff
RUN apt-get update && \
    apt-get install -y python3-pip wget git   

    # Pytorch
RUN wget https://nvidia.box.com/shared/static/06vlvedmqpqstu1dym49fo7aapgfyyu9.whl -O torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl && \
    pip3 install numpy torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl && \
    rm torch-1.2.0a0+8554416-cp36-cp36m-linux_aarch64.whl

    # torchvision
RUN apt-get install -y libjpeg-dev zlib1g-dev libpng-dev python3-matplotlib && \
    pip3 install Pillow torchvision

# OpenCV from source
RUN apt-get install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev && \
    apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
    apt-get install -y python2.7-dev && \
    apt-get install -y python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev && \
    apt-get install -y libv4l-dev v4l-utils qv4l2 v4l2ucp && \
    apt-get install -y curl unzip && \
    apt-get update

RUN curl -L https://github.com/opencv/opencv/archive/3.4.0.zip -o opencv-3.4.0.zip && \
    unzip opencv-3.4.0.zip

RUN cd opencv-3.4.0/ && \
    mkdir release && \
    cd release/ && \
    cmake -D WITH_CUDA=ON -D CUDA_ARCH_BIN="6.2" -D CUDA_ARCH_PTX="" -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make && \
    make install

# SiamMask requirements
RUN apt-get install -y python3-h5py python3-scipy python3-llvmlite python3-pandas python3-numba && \
    pip3 install Cython==0.29.4 colorama==0.3.9 requests==2.21.0 fire==0.1.3 tqdm==4.29.1 tensorboardX==1.6

# SiamMask itself
RUN git clone https://github.com/foolwood/SiamMask.git && cd SiamMask && \
    export SiamMask=$PWD && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    bash make.sh && \
    export PYTHONPATH=$PWD:$PYTHONPATH
ENV SiamMask=/SiamMask
ENV PYTHONPATH=/SiamMask:$PYTHONPATH
# Download models
RUN cd $SiamMask/experiments/siammask_sharp && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth && \
    wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth

