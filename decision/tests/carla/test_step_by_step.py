import sys, os
sys.path.append("../../")
from ensemble import VehicleController, LocalPlannerType, PhysicalParameters, PlanningPipeline, Interpolator, MotionController, PlannerResultType
import time
from pydriveless import WorldPose, MapPose, Waypoint, CoordinateConverter, angle
from carla_test_utils import read_path, init_sim
import cv2
from carladriver import CarlaEgoVehicle, CarlaSimulation, CarlaSLAM, BevCameraSemantic
from pydriveless import Telemetry
import json
from ensemble import Ensemble

GPS_PERIOD_MS=100
IMU_PERIOD_MS=100

###
## Step by step tester
###

VELOCITY = 5

in_motion_driving = False

def step_calibrate(sim: CarlaSimulation, ego: CarlaEgoVehicle, slam: CarlaSLAM, camera: BevCameraSemantic) -> PlanningPipeline:
    gps = ego.attach_gps_sensor(GPS_PERIOD_MS)
    imu = ego.attach_imu_sensor(IMU_PERIOD_MS)
        
    data = gps.read()
    while not data.valid:
        data = gps.read()
        pass

    data = imu.read()
    while not data.valid:
        data = imu.read()
        pass

    data = camera.read()
    while data is None:
        data = camera.read()
        pass

    gps_data = gps.read()
    imu_data = imu.read()
    origin = WorldPose(
            lat=angle.new_deg(gps_data.lat),
            lon=angle.new_deg(gps_data.lon),
            alt=gps_data.alt,
            compass=angle.new_deg(imu_data.compass)
        )

    return PlanningPipeline(origin)

def step_read_next_goals(slam, path, path_pos):
    pos = MapPose.find_nearest_goal_pose(
            location=slam.estimate_ego_pose(),
            poses=path,
            start=path_pos,
            max_hopping=5
        )   
    hopping_distance_to_end = (path_pos - len(path))
    if pos < 0:
        if hopping_distance_to_end > 1:
            print ("invalid self position")
            return None, None
        else:
            print ("finish driving")
            return None, None

    if pos == len(path) - 1:
        g1, g2 = path[-1], None
    else:
        g1, g2 = path[pos], path[pos + 1]

    return g1, g2

def step_build_planning_data(seq: int, slam: CarlaSLAM, camera: BevCameraSemantic, planning_pipeline: PlanningPipeline, g1: MapPose, g2: MapPose):
    location = slam.estimate_ego_pose()
    frame, ts = camera.read()
    return planning_pipeline.step1_build_planning_data(
            seq=seq,
            bev=frame,
            ego_location=location,
            g1=g1,
            g2=g2,
            velocity=VELOCITY
        )

def on_finished_motion(self, mc) -> None:
    mc.cancel()
    mc.brake()
    in_motion_driving = False
    print ("finished motion")

sim, ego, slam = init_sim()
time.sleep(2)
path = read_path("test_motion_controller_goal_points.dat")
sim.show_path(path)

cam = ego.init_semantic_bev_camera()

planning_pipeline = step_calibrate(sim, ego, slam, cam)
#local_planner = Ensemble(planning_pipeline.get_coord_converter(), max_exec_time_ms=8000)
local_planner = Interpolator(planning_pipeline.get_coord_converter(), max_exec_time_ms=8000)

motion_controller = MotionController(
        period_ms=2,
        longitudinal_controller_period_ms=50,
        ego=ego,
        slam=slam,
        odometer=ego.get_odometer_sensor()
    )
motion_controller.start()


drive_path = True
path_pos = 0
while drive_path:
    motion_controller.brake()
    g1, g2 = step_read_next_goals(slam, path, path_pos)
    if g1 is None: 
        drive_path = False
        continue

    print (f"driving to g1 = {g1}, g2 = {g2}")

    
    planning_data = step_build_planning_data(path_pos + 1, slam, cam, planning_pipeline, g1, g2)
    planning_pipeline.step3_pre_process(planning_data)

    if not planning_pipeline.step4_find_local_goal(planning_data):
        print (f"unable to find local goal for {g1}, {g2}")
        Telemetry.log("log/error_planning.log", planning_data)
        Telemetry.log("log/error_planning_bev.png", planning_data.og())
        Telemetry.log("log/error_planning_bevc.png", planning_data.og().get_color_frame())
        drive_path = False
    
    planning_pipeline.step5_perform_local_planning(planning_data, local_planner)

    while local_planner.is_planning():
        pass
        
    if local_planner.timeout():
        print ("timeout")
        Telemetry.log("log/timeout_planning.log", planning_data)
        Telemetry.log("log/timeout_planning_bev.png", planning_data.og())
        Telemetry.log("log/timeout_planning_bevc.png", planning_data.og().get_color_frame())
        input()
        continue

    res = local_planner.get_result()

    if res.result_type != PlannerResultType.VALID:
        print ("invalid path")
        path_pos += 1
        Telemetry.log("log/invalid_planning.log", planning_data)
        Telemetry.log("log/invalid_planning_bev.png", planning_data.og())
        Telemetry.log("log/invalid_planning_bevc.png", planning_data.og().get_color_frame())
        input()
        continue

    map_path = planning_pipeline.step6_translate_local_path_to_map_coordinates(planning_data, res)
    sim.show_path(map_path)

    # drv_path = [str(p) for p in map_path]
    # Telemetry.log("driving_planned_path.json", json.dumps(drv_path))

    print ("finished planning. Hit enter to drive")
    Telemetry.log(f"log/bev_{path_pos}.log", planning_data)
    Telemetry.log(f"log/local_path_{path_pos}.log", res.path)
    input()

    print ("driving...")

    # path_str = [str(p) for p in map_path]
    # json_path = json.dumps(path_str)
    # Telemetry.log("log/motion_controller.log", json_path, append=True)
    # while not Telemetry.empty():
    #     time.sleep(0.1)

    motion_controller.set_path(map_path, velocity=2)
    while motion_controller.is_tracking():
        time.sleep(0.1)

    motion_controller.brake()    
    # while in_motion_driving:
    #     pass

    print("vehicle stopped")
    input()
    #sim.clear_paths()
    path_pos += 1

ego.destroy()
