FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 1. 基础工具与 ROS 2 仓库配置
RUN apt-get update && apt-get install -y \
    locales curl gnupg2 lsb-release git cmake build-essential \
    python3-pip python3-rosdep python3-vcstool \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu \
    $(. /etc/os-release && echo $VERSION_CODENAME) main" \
    | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 2. 安装 ROS 2 + 常用开发工具
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-argcomplete \
    # 常用ROS依赖
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-pcl-ros \
    ros-humble-tf2-tools \
    ros-humble-rviz2 \
    ros-humble-serial-driver \
    ros-humble-asio-cmake-module \
    # 系统库
    libeigen3-dev libboost-all-dev libpcap-dev libusb-1.0-0-dev libproj-dev \
    libflann-dev libqhull-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libjpeg-dev libpng-dev libtiff-dev libopenexr-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. 初始化 rosdep（关键！！！）
RUN rosdep init && rosdep update

# 4. OpenCV
WORKDIR /opt
RUN git clone --branch 4.6.0 --depth 1 https://github.com/opencv/opencv.git && \
    git clone --branch 4.6.0 --depth 1 https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
          -D WITH_GTK=ON \
          -D WITH_OPENGL=ON \
          -D WITH_TBB=ON \
          -D BUILD_EXAMPLES=OFF .. && \
    make -j$(nproc) && make install && cd /opt && rm -rf opencv*

# 5. PCL
RUN git clone --branch pcl-1.12.1 --depth 1 https://github.com/PointCloudLibrary/pcl.git && \
    cd pcl && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_VTK=ON \
          -D WITH_OPENGL=ON \
          -D BUILD_visualization=ON \
          -D BUILD_examples=OFF .. && \
    make -j$(nproc) && make install && cd /opt && rm -rf pcl

# 6. 创建工作空间 + serial_driver（关键）
WORKDIR /home/workspace
RUN mkdir -p /home/workspace/src && \
    cd /home/workspace/src && \
    git clone https://github.com/ros-drivers/transport_drivers.git

# 7. 自动source（非常关键）
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /home/workspace/install/setup.bash" >> /root/.bashrc

# 8. 默认工作目录
WORKDIR /home/workspace