#!/bin/bash

PROJECT_ROOT="/home/duang/vision"
CONTAINER_NAME="robot_vision"

# 允许 docker 访问 X11
xhost +local:docker > /dev/null

# Wayland 兼容（很关键）
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export WAYLAND_DISPLAY=$WAYLAND_DISPLAY

# 如果容器存在 → 启动
if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
    echo "Starting existing container..."
    docker start ${CONTAINER_NAME}

    docker exec -it \
        -e DISPLAY=$DISPLAY \
        -e QT_QPA_PLATFORM=xcb \
        ${CONTAINER_NAME} /bin/bash

else
    echo "Creating new container..."

    docker run -it \
        --name ${CONTAINER_NAME} \
        --gpus all \
        --privileged \
        --network host \
        --shm-size=8g \
        \
        -e DISPLAY=$DISPLAY \
        -e QT_X11_NO_MITSHM=1 \
        -e QT_QPA_PLATFORM=xcb \
        -e NVIDIA_DRIVER_CAPABILITIES=all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v $XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR \
        \
        -v ${PROJECT_ROOT}:/home/workspace/vision \
        --workdir /home/workspace \
        \
        robotics_image:latest \
        /bin/bash
fi
