#! /bin/sh
cd /home/out0/libdrivelessfw && make && sudo make install
cd /home/out0/libvehiclehal && make carla && sudo make install