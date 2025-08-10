import sys, os
sys.path.append("../../")
from ensemble import VehicleController, LocalPlannerType, PhysicalParameters
import time
from pydriveless import MapPose, CoordinateConverter
from carla_test_utils import read_path, init_sim
import cv2

GPS_PERIOD_MS=100
IMU_PERIOD_MS=100

###
## This test executes the full state-machine Vehicle controller, with interpolator Local Planner
###

sim, ego, slam = init_sim()
path = read_path("test_motion_controller_goal_points.dat")
sim.show_path(path)

cam = ego.init_semantic_bev_camera()

controller = VehicleController(
        vehicle=ego,
        gps=ego.attach_gps_sensor(period_ms=GPS_PERIOD_MS),
        imu=ego.attach_imu_sensor(period_ms=IMU_PERIOD_MS),
        input_camera=cam,
        odometer=ego.get_odometer_sensor(),
        slam=slam,
        local_planner_timeout_ms=-1,
        local_planner_type=LocalPlannerType.INTERPOLATOR,
        sim=sim,
    )

try:

    controller.start()

    time.sleep(1)
    controller.drive(path)

    print ("Press <enter> to stop")
    input()
finally: 
    controller.destroy()
    ego.destroy()
