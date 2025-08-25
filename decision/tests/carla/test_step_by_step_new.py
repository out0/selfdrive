from ensemble import PlanningPipeline, Interpolator, LocalPlannerExecutor, Ensemble, MotionController, PlannerResultType, PlanningData, PlanningResult
import time
from pydriveless import WorldPose, MapPose, Waypoint, angle
from carla_test_utils import read_path, init_sim
from carladriver import CarlaEgoVehicle, CarlaSimulation, CarlaSLAM, BevCameraSemantic
from pydriveless import Telemetry
import os
GPS_PERIOD_MS=100
IMU_PERIOD_MS=100
VELOCITY=2
###
## Step by step tester
###
class SimulationData:
    ego : CarlaEgoVehicle
    sim: CarlaSimulation
    slam: CarlaSLAM
    path: list[MapPose]
    cam: BevCameraSemantic
    pipeline: PlanningPipeline
    motion_controller: MotionController
    planning_data: PlanningData
    planning_result: PlanningResult
    path_pos: int
    local_planner: LocalPlannerExecutor
    drive_path: list[MapPose]

    def __init__(self, ego: CarlaEgoVehicle, sim: CarlaSimulation, slam: CarlaSLAM, path: list[MapPose]):
        self.ego = ego
        self.sim = sim
        self.slam = slam
        self.path = path
        self.planning_data = None
        self.planning_result = None
        self.path_pos = 0
        self.drive_path = None
        pass

def menu():
    print ("driving options: ")
    print ("")
    print ("1. Compute path position / goals")
    print ("2. Plan local path")
    
    print ("3. Drive")
    print ("4. Set path pos")
    print ("5. Quit")
    return input().replace("\n", "")

    # fd = sys.stdin.fileno()
    # old_settings = termios.tcgetattr(fd)
    # try:
    #     tty.setraw(fd)
    #     ch = sys.stdin.read(1)
    # finally:
    #     termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    # return ch

def start_simulation(path_file: str):
    sim, ego, slam = init_sim()
    time.sleep(2)
    path = read_path(path_file)
    sim.show_path(path)
    return SimulationData(ego, sim, slam, path)

def step_calibrate(sim_data: SimulationData) -> None:
    gps = sim_data.ego.attach_gps_sensor(GPS_PERIOD_MS)
    imu = sim_data.ego.attach_imu_sensor(IMU_PERIOD_MS)
    camera = sim_data.ego.init_semantic_bev_camera()
        
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

    sim_data.pipeline = PlanningPipeline(origin)

    sim_data.motion_controller = MotionController(
            period_ms=2,
            longitudinal_controller_period_ms=50,
            ego=sim_data.ego,
            slam=sim_data.slam,
            odometer=sim_data.ego.get_odometer_sensor()
        )
    sim_data.cam = camera
    sim_data.motion_controller.start()
    sim_data.motion_controller.brake()
    sim_data.path_pos = 0

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

def str_mp(p: MapPose) -> str:
    return f"({p.x:.2f}, {p.y:.2f}, {p.heading.deg():.2f})"
def str_wp(p: Waypoint) -> str:
    return f"({p.x}, {p.z}, {p.heading.deg():.2f})"

def menu_opt_compute_goals(sim_data: SimulationData) -> None:
    sim_data.g1 = None
    sim_data.g2 = None
    sim_data.g1, sim_data.g2 = step_read_next_goals(sim_data.slam, sim_data.path, sim_data.path_pos)
    if sim_data.g1 is None: 
        print ("g1 not found")
        return
    epos = sim_data.slam.estimate_ego_pose()
    print (f"driving to g1 = {str_mp(sim_data.g1)}, g2 = {str_mp(sim_data.g2)}")
    print (f"current pos: {str_mp(epos)}")

    conv = sim_data.pipeline.get_coord_converter()

    sim_data.planning_data = step_build_planning_data(sim_data.path_pos, sim_data.slam, sim_data.cam, sim_data.pipeline, sim_data.g1, sim_data.g2)
    
    sim_data.pipeline.step3_pre_process(sim_data.planning_data)

    if not sim_data.pipeline.step4_find_local_goal(sim_data.planning_data):
        print (f"unable to find local goal for g1 = {str_mp(sim_data.g1)}, g2 = {str_mp(sim_data.g2)}")
        Telemetry.log("log/error_planning.log", sim_data.planning_data)
        Telemetry.log("log/error_planning_bev.png", sim_data.planning_data.og())
        Telemetry.log("log/error_planning_bevc.png", sim_data.planning_data.og().get_color_frame())

    print (f"g1 --> L1 = {conv.convert(sim_data.planning_data.ego_location(), sim_data.g1)}, selected local goal {str_wp(sim_data.planning_data.local_goal())}")
    print ("\n\n")

def menu_opt_lp(sim_data: SimulationData) -> None:
    if sim_data.planning_data is None:
        print ("please build the planning data first\n\n")
        return
    
    sim_data.pipeline.step5_perform_local_planning(sim_data.planning_data, sim_data.local_planner)

    while not sim_data.local_planner.new_path() and sim_data.local_planner.is_planning():
        time.sleep(0.01)
        
    if not sim_data.local_planner.new_path():
        print ("Invalid path\n\n")
        return

    res = sim_data.local_planner.get_result()

    if res is None:
        return

    sim_data.local_planner.cancel()
       
    if sim_data.local_planner.timeout():
        print ("timeout\n\n")
        Telemetry.log("log/timeout_planning.log", sim_data.planning_data)
        Telemetry.log("log/timeout_planning_bev.png", sim_data.planning_data.og())
        Telemetry.log("log/timeout_planning_bevc.png", sim_data.planning_data.og().get_color_frame())
        return

    if res.result_type != PlannerResultType.VALID:
        print ("invalid path\n\n")
        #path_pos += 1
        Telemetry.log("log/invalid_planning.log", sim_data.planning_data)
        Telemetry.log("log/invalid_planning_bev.png", sim_data.planning_data.og())
        Telemetry.log("log/invalid_planning_bevc.png", sim_data.planning_data.og().get_color_frame())
        return
        
    map_path = sim_data.pipeline.step6_translate_local_path_to_map_coordinates(sim_data.planning_data, res)
    sim_data.sim.show_path(map_path)
    sim_data.drive_path = map_path
    sim_data.planning_result = res
    print (f"planned path with {len(map_path)} points, planner: {res.planner_name}, exec time: {res.planning_exec_time_ms}")
    print ("\n\n")

SD = start_simulation("test_motion_controller_goal_points.dat")
step_calibrate(SD)
SD.local_planner = Ensemble(SD.pipeline.get_coord_converter(), max_exec_time_ms=-1)

def menu_drive(sim_data: SimulationData) -> None:
    if sim_data.drive_path is None:
        print ("please plan the path first\n\n")
        return
    Telemetry.log(f"log/bev_{sim_data.path_pos}.log", sim_data.planning_data)
    Telemetry.log(f"log/local_path_{sim_data.path_pos}.log", sim_data.planning_result.path)
    sim_data.motion_controller.set_path(sim_data.drive_path, velocity=2)
    while sim_data.motion_controller.is_tracking():
        time.sleep(0.05)
    sim_data.motion_controller.brake()

run = True
while run:
    choice = menu()
    os.system("clear")
    match choice:
        case '1':
            menu_opt_compute_goals(SD)
        case '1l':
            for i in range(20):
                menu_opt_compute_goals(SD)
                time.sleep(0.1)
        case '2':
            if SD.sim.count_paths() > 1:
                SD.sim.clear_last_path()
            menu_opt_lp(SD)
            pass
        case '2l':
            for i in range(20):
                if SD.sim.count_paths() > 1:
                    SD.sim.clear_last_path()
                menu_opt_lp(SD)
            pass
        case '3':
            menu_drive(SD)
            pass
        case '4':
            pass
        case '5':
            run = False
            pass

SD.motion_controller.destroy()
SD.ego.destroy()
SD.sim.clear_paths()
SD.sim.clear_objects()
SD.sim.clear_coordinates()
SD.sim.destroy()