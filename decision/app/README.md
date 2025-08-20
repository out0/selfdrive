# Ensemble 

## Intro

The Ensemble python module comprises all basic Path Planning tasks to guide the EGO in an unstructued scenario. It is subdivided into model, motion, planner and slam. 

- **model module**: Implements basic curve quality measurements, the planner executor interface, and classes to hold planning data and results.
- **motion module**: Implements lateral and longitudinal controllers.
- **planner module**: Implements local planners, including the Ensemble and its sub-executors: Hybrid A*, Interpolator, Overtaker, and Bi-RRT*.
- **control module**: Provides the basic planning pipeline and state machine to integrate data acquisition, pre-processing, planning, and motion.

## Control

VehicleController class implements the basic EGO controller. The EGO car is defined by EgoVehicle, an interface to basic control implementation (accel, brake, steering, etc.). 

### Basic setup
```python 

path: list[MapPose] = ...
gps_sensor: GPS  = ...
imu_sensor: IMU = ...
camera_sensor: Camera = ...
odometer_sensor: Odometer= ...

controller = VehicleController( 
        vehicle=ego,  
        gps=gps_sensor,  
        imu=imu_sensor,  
        input_camera=camera_sensor,  
        odometer=odometer_sensor,  
        slam=slam,  
        local_planner_timeout_ms=500,  
        local_planner_type=LocalPlannerType.ENSEMBLE  
    ) 

controller.start()  
controller.drive(path)
``` 



