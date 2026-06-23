# POC: Autonomous Car running in Carla Simulator

Proof-of-concept of planning aproaches for running a car in a simulated environment.

This is a research repo for testing planning theories.

Main features:

- carla_driver: Implements a driver to the Carla Simulator. The simulator basic controls are abstracted as CarlaEgoVehicle, which implements EgoVehicle, an interface that represents any Autonomous Vehicle being controlled. This module also provides carla implementations to many sensors such as GPS, IMU and cameras.

- decision: The decision-making layer. Here we implement the vehicle motion control, local planning and behavior control state machine.
[Local Path Planning for Self-driving Cars in Unknown Environments Using an Ensemble-Based Approach](https://ieeexplore.ieee.org/document/11420592)


- libdriveless - Implements the basic features used in this project: basic coordinate elements Waypoint, MapPose, WorldPose, Quaternion, State, CUDA frame for parallel GPU processing, CPU frame for parallel CPU processing, Search Frame for Occupancy Grid parallel processing (CPU and GPU versions).

- FastRRT - Implements our proposal for Fast RRT planning (article under publication process)

- libgpd - Implements our proposal for Local goal planning discover, to allow continuous vehicle navigation in unknown environments.
[Local Path Planning for Self-driving Cars in Unknown Environments Using an Ensemble-Based Approach](https://ieeexplore.ieee.org/document/11420592)

- dev_container - Contains basic configuration to build a local dev environment

- datalink - Datalink provides fast dual easy client-server connection between two terminals A <------> B

- libvision - C++ library with python bindings that implements basic vision features for self-driving projects.
# FastRRT
