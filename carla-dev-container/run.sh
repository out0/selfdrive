#! /bin/sh

docker run --gpus all \
    --privileged \
    --rm \
    --net host \
    --ipc host \
    --env DISPLAY=$DISPLAY \
    --env XAUTHORITY=$XAUTHORITY \
    --env PULSE_SERVER=unix:/run/user/1000/pulse/native \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --volume /dev/video0:/dev/video0 \
    --volume /home/cristiano/.Xauthority:/home/out0/.Xauthority:rw \
    --volume /home/cristiano/Documents/Projects:/home/out0 \
    --runtime nvidia \
    -e SDL_HINT_CUDA_DEVICE=0 \
    -e NVIDIA_VISIBLE_DEVICES=0 driveless-dev-cuda-img
