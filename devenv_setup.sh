#!/bin/bash

set -e

sudo apt install -y  build-essential pkg-config libglib2.0-dev libgsl-dev

sudo rm -rf /tf
sudo mkdir -p /tf
sudo chmod a+rw /tf
sudo apt install -y git cmake libssl-dev
cd /tf && wget https://github.com/Kitware/CMake/releases/download/v4.4.2/cmake-4.4.2.tar.gz && tar xvf cmake-4.4.2.tar.gz
mkdir -p /tf/cmake-4.4.2/build && cd /tf/cmake-4.4.2/build && cmake ../ && make -j$(nproc) && sudo make install
sudo rm -rf /tf

sudo rm -rf /tf
sudo mkdir -p /tf
sudo chmod a+rw /tf
cd /tf && git clone https://github.com/oneapi-src/oneTBB.git
sudo mkdir -p /tf/oneTBB/build
cd /tf/oneTBB/build && cmake .. && make -j$(nproc) && sudo make install

sudo apt install -y python3-numpy
sudo rm -rf /tf
sudo mkdir -p /tf
sudo chmod a+rw /tf
cd /tf &&  git clone https://github.com/opencv/opencv.git && git clone https://github.com/opencv/opencv_contrib.git
mkdir -p /tf/opencv/build
cd /tf/opencv && git checkout 5.0.0 && cd /tf/opencv_contrib && git checkout 5.0.0
sudo rm -rf /usr/include/numpy
sudo ln -s /usr/lib/python3/dist-packages/numpy/_core/include/numpy /usr/include/numpy

read -p "Enable CUDA? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing CUDA dependencies..."

    wget -c https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.23.1.3_cuda13-archive.tar.xz
    tar xvf ./cudnn-linux-x86_64-9.23.1.3_cuda13-archive.tar.xz
    sudo mv ./cudnn-linux-x86_64-9.23.1.3_cuda13-archive/lib/* /usr/local/cuda/lib64/
    sudo mv ./cudnn-linux-x86_64-9.23.1.3_cuda13-archive/include/* /usr/local/cuda/include
    sudo rm -rf ./cudnn-linux-x86_64-9.23.1.3_cuda13-archive
    sudo ldconfig -v
    rm -rf ./cudnn-linux-x86_64-9.23.1.3_cuda13-archive.tar.xz

    cd /tf/opencv/build && sudo cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/tf/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF .. \
    -D WITH_OPENMP=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_OPENCL=OFF \
    -D BUILD_ZLIB=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D WITH_GTK=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_VERSION=314 \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.14 \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3.14 \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.14 \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.14/dist-packages/numpy/core/include \
    -D WITH_QT=OFF \
    -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.14/dist-packages \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.14/site-packages/ \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_CUDA=ON \
    -D CUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so \
    -D CUDNN_INCLUDE_DIR=/usr/local/cuda/include \
    -D BUILD_opencv_cudacodec=ON \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D CUDA_ARCH_BIN=8.6 \
    -D WITH_CUBLAS=1
else
    echo "Skipping CUDA installation."
    cd /tf/opencv/build && sudo cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/tf/opencv_contrib/modules \
    -D BUILD_EXAMPLES=OFF .. \
    -D WITH_OPENMP=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_OPENCL=OFF \
    -D BUILD_ZLIB=ON \
    -D BUILD_TIFF=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_TBB=ON \
    -D BUILD_TESTS=OFF \
    -D WITH_EIGEN=OFF \
    -D WITH_GSTREAMER=ON \
    -D WITH_V4L=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_VTK=OFF \
    -D WITH_GTK=ON \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_VERSION=314 \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3.14 \
    -D PYTHON3_EXECUTABLE=/usr/bin/python3.14 \
    -D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
    -D PYTHON3_INCLUDE_DIR=/usr/include/python3.14 \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/local/lib/python3.14/dist-packages/numpy/_core/include \
    -D WITH_QT=OFF \
    -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.14/dist-packages \
    -D PYTHON3_PACKAGES_PATH=/usr/local/lib/python3.14/site-packages/ \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_GENERATE_PKGCONFIG=ON
fi

cd /tf/opencv/build && sudo make -j$(nproc) && sudo make install && sudo rm -rf /tf

git clone https://github.com/google/googletest
cd googletest && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install
rm -rf googletest
