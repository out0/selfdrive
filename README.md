# POC: Autonomous Car running in Carla Simulator

Proof-of-concept of planning aproaches for running a car in a simulated environment.

This is a research repo for testing planning theories.

Main features:

- carla_driver: Implements a driver to the Carla Simulator. The simulator basic controls are abstracted as CarlaEgoVehicle, which implements EgoVehicle, an interface that represents any Autonomous Vehicle being controlled. This module also provides carla implementations to many sensors such as GPS, IMU and cameras.

- decision: The decision-making layer. Here we implement the vehicle motion control, local planning and behavior control state machine.

- libdriveless - Implements the basic features used in this project: basic coordinate elements Waypoint, MapPose, WorldPose, Quaternion, State and CUDA frame for parallel GPU processing.

- libfastrrt - Implements our proposal for Fast RRT planning (article under review)

- libgpd - Implements our proposal for Local goal planning discover, to allow continuous vehicle navigation in unknown environments  (article under review).

- dev_container - Contains basic configuration to build a local dev environment
