# POC: Autonomous Car running in Carla Simulator

Proof-of-concept of planning aproaches for running a car in a simulated environment.

This is a research repo for testing planning theories.

The idea is to put this code into an already built vehicle called The Crawler, for field testing.

Main features:

- Vision: a BEV frame is simulated by putting a camera above the vehicle. This is a temporary resource for allowing the continuity of the research.
          We're currently using the semantic segmentation feature from the simulation platform rather than using neural networks to predict classes, but
          it will change very soon in order to reflect real world behavior. 

- Global planning: currently using a stub which provides a fixed global goal list

- Local planning: a fully implemented controller, capable of deciding local goal waypoints, using a local planner approach such as A* (currently implemented) and self-location in the planned mission 

- Motion execution: a ACC usind PID controller and a Stanley lateral controller were implemented.

- Behavior control: the AV is capable of planning and replanning and will do it's best to fulfill the mission. If it can't do it, it will
  notify upper layers, which can perform a global replanning or warning the human operator to takeover the AV.

- Manual control: a manual controller is implemented and integrated with an API service for using external tools such as an android app for controlling the
virtual car, if needed. A small command-line tool can also be used to manually control the AV.

