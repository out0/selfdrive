#! /bin/sh
# cp -R ../../driveless/libvehiclehal .
# cp -R ../../driveless/libdrivelessfw .
# cp -R ../utils/cudac .
# cp -R ../utils/datalink .
docker build -t out0/driveless-dev-cuda -f img/Dockerfile-simple .
# rm -rf cudac
# rm -rf datalink

