#! /bin/sh
cp -R ../../../libvehiclehal .
cp -R ../../../libdrivelessfw .
docker build -t driveless-dev18-cuda-img -f img/Dockerfile-ubuntu18 .
rm -rf libvehiclehal
rm -rf libdrivelessfw