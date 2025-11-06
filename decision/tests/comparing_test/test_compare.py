import math
from pydriveless import WorldPose, MapPose, Waypoint, CoordinateConverter, PI
from pydriveless import SearchFrame, angle, SearchParams, EgoParams
from pyfastrrt import FastRRT
from ensemble import LocalPlannerExecutor, PlanningData, PlanningResult, PlannerResultType, PhysicalParameters, Ensemble, InformedHybridAStar
# planners
from ensemble import FastRRTPlanner, HybridAStar, Interpolator
import numpy as np
import math
import cv2
import time
#
# Test suit
#
EGO_DIMENSIONS_PX = (20, 40)
ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))


def mat_trans(x: int, z: int) -> np.ndarray:
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [x, z, 1]
    ])


def add_ego(frame, start: Waypoint) -> None:
    hh = EGO_DIMENSIONS_PX[1] // 2
    ww = EGO_DIMENSIONS_PX[0] // 2

    c = math.cos(start.heading.rad())
    s = math.sin(start.heading.rad())
    x = start.x
    z = start.z

    Mr = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    M = mat_trans(-x, -z) @ Mr @ mat_trans(x, z)

    for j in range(-hh, hh):
        for i in range(-ww, ww):
            p = np.array([x+i, z+j, 1]) @ M
            # zp = z + j
            # xp = x + i
            xp = int(p[0])
            zp = int(p[1])
            if xp < 0 or xp >= frame.shape[1]:
                continue
            if zp < 0 or zp >= frame.shape[0]:
                continue
            frame[zp, xp] = [255, 0, 0]

    draw_arrow(frame, x, z, start.heading, arrow_length=50)


def draw_arrow(frame: np.ndarray, x: int, z: int, heading: angle, arrow_length=20):
    # Arrow end point
    a = heading.rad() - (PI / 2)
    end_x = int(x + arrow_length * math.cos(a))
    end_z = int(z + arrow_length * math.sin(a))

    # Draw the arrow shaft (line)
    cv2.line(frame, (x, z), (end_x, end_z), (0, 0, 255), thickness=2)
    # Draw arrow head (simple triangle)
    angle = math.atan2(end_z - z, end_x - x)
    arrow_head_size = 8
    for side in [-1, 1]:
        side_angle = angle + side * math.radians(25)
        hx = int(end_x - arrow_head_size * math.cos(side_angle))
        hy = int(end_z - arrow_head_size * math.sin(side_angle))
        cv2.line(frame, (end_x, end_z), (hx, hy), (0, 0, 255), thickness=2)


def convert_black_white_frame(frame: np.ndarray) -> np.ndarray:
    f = np.zeros(frame.shape, dtype=np.float32)
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            f[i, j, 0] = 1.0 if frame[i, j, 0] == 255 else 0
            # f[i, j, 0] = 1.0
    return f

    # map_center_location = MapPose(0, 0, 0, angle.new_rad(0))
    # ego_location = conv.convert(map_center_location, start)
    # l0 = conv.convert(map_center_location, ego_location)
    # g1 = conv.convert(map_center_location, pose=goal)
    # g2 = None


def exec_planner_test(outp_frame: np.ndarray, search_params: SearchParams, executor: LocalPlannerExecutor, path_color: tuple[int, int, int] = [255, 0, 0]) -> None:
    print(f"Starting planner test for {executor.get_planner_name()}")
    executor.plan(search_params, True)

    while not executor.new_path_available():
        time.sleep(0.1)
    result: PlanningResult = executor.get_result()

    if result.result_type != PlannerResultType.VALID:
        print(f"{executor.get_planner_name()} failed to find a valid path.")
        return

    execution_time = executor.get_execution_time()
    print(
        f"{executor.get_planner_name()} execution time: {execution_time:.2f} ms [choosen: {result.planner_name}]")
    path = result.path

    if executor.is_optimizing():
        print(f"Planner {executor.get_planner_name()} is optimizing")
        while not executor.is_optimizing():
            time.sleep(0.1)

        execution_time = executor.get_execution_time()
        print(
            f"{executor.get_planner_name()} optimizing execution time: {execution_time:.2f} ms [choosen: {result.planner_name}]")
        path = result.path

    for p in path:
        outp_frame[p.z, p.x, :] = path_color


def exec_test():

    start = Waypoint(436, 250, angle.new_deg(0))
    goal = Waypoint(48, 261, angle.new_deg(-180))

    # start = Waypoint(455, 263, angle.new_deg(0))
    # goal =  Waypoint(48, 261, angle.new_deg(0))
    # goal =  Waypoint(207, 117, angle.new_deg(-180))

    orig_frame = np.array(cv2.imread("comparing_og.png"))
    raw_frame = convert_black_white_frame(orig_frame)
    width, height = raw_frame.shape[1], raw_frame.shape[0]
    lower_bound = (
        start.x - EGO_DIMENSIONS_PX[0], start.z + EGO_DIMENSIONS_PX[1])
    upper_bound = (
        start.x + EGO_DIMENSIONS_PX[0], start.z - EGO_DIMENSIONS_PX[1])
    # px, pz = width*PhysicalParameters.OG_WIDTH_PX_TO_METERS_RATE, height*PhysicalParameters.OG_HEIGHT_PX_TO_METERS_RATE
    px, pz = width, height

    ego_params = EgoParams.init(width, height)\
        .with_ego_lower_bound(lower_bound)\
        .with_ego_upper_bound(upper_bound)\
        .with_max_steering_angle(angle.new_deg(40))\
        .with_max_curvature(0.34)\
        .with_segmentation_class_costs(np.array([-1.0, 0.0]))\
        .with_segmentation_class_colors(np.array([[0, 0, 0], [255, 255, 255]]))\
        .with_search_physical_size(px, pz)\
        .with_vehicle_length(PhysicalParameters.VEHICLE_LENGTH_M)\
        .build()

    frame = ego_params.new_search_frame()
    frame.set_frame_data(raw_frame)
    frame.process_distance_to_goal(goal.x, goal.z)
    frame.process_safe_distance_zone(EGO_DIMENSIONS_PX, True)

    search_params = ego_params.new_search_params(start, goal)\
        .with_distance_to_goal_tolerance(20.0)\
        .with_frame(frame)\
        .with_max_path_size(40.0)\
        .with_min_distance((20, 40))\
        .with_velocity(1.0)\
        .with_distance_to_goal_tolerance(5)\
        .with_timeout(3000)\
        .build()

    # fast_rrt_planner = FastRRTPlanner(ego_params, False, True)
    # exec_planner_test(orig_frame, search_params, fast_rrt_planner, path_color=[255, 0, 0])

    has = HybridAStar(ego_params)
    exec_planner_test(orig_frame, search_params, has, path_color=[0, 0, 255])

    # interpolator = Interpolator(ego_params)
    # exec_planner_test(orig_frame, search_params,
    #                   interpolator, path_color=[128, 0, 128])

    add_ego(orig_frame, start)

    draw_arrow(orig_frame, goal.x, goal.z, goal.heading)

    cv2.imwrite("debug.png", orig_frame)


if __name__ == "__main__":
    exec_test()
