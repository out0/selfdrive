
from .waypoint import Waypoint, angle
from .world_pose import WorldPose 
from .map_pose import MapPose
from .search_params import EgoParams, SearchParams
from . search_frame import SearchFrame
from .search_frame_cpu import SearchFrameCPU
import time
import json, numpy as np
import os, cv2

PFM = 1
RGB = 2

class TestConfig:
    start: Waypoint
    goal: Waypoint
    file_type: int
    raw_frame: np.ndarray
    segmentation_costs: list[int]
    segmentation_colors: list[tuple[int]]
    lower_bound: tuple[int, int]
    upper_bound: tuple[int, int]
    og_real_size: tuple[float, float]
    max_curvature: float
    max_steering_angle_deg: float
    vehicle_length_m: float
    meters_to_pixel_ratio: tuple[float, float]
    pixel_to_meters_ratio: tuple[float, float]
    world_origin: WorldPose

class TestUtils:
    def __init__(self):
        pass

    def __read_pfm(file_path):
        with open(file_path, 'rb') as f:
            header = f.readline().decode('utf-8').rstrip()
            if header == 'PF':
                color = True
            elif header == 'Pf':
                color = False
            else:
                raise ValueError("Not a PFM file.")

            dims_line = ''
            while True:
                line = f.readline().decode('utf-8')
                if line.startswith('#'):
                    continue  # skip comments
                dims_line = line
                break
            width, height = map(int, dims_line.strip().split())

            scale = float(f.readline().decode('utf-8').strip())
            endian = '<' if scale < 0 else '>'  # little endian if scale < 0
            scale = abs(scale)

            data = np.fromfile(f, endian + 'f')
            shape = (height, width, 3) if color else (height, width)

            data = np.reshape(data, shape)
            data = np.flipud(data)  # PFM stores pixels from bottom to top

            return data
        
    def __convert_pfm(raw) -> np.ndarray:
        new_frame = np.zeros((raw.shape[0], raw.shape[1], 3), dtype=np.float32)
        
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                if np.isfinite(raw[i, j]):  # unknown == Obstacle
                    new_frame[i, j] = [1.0, 255*float(raw[i, j])/0.75, 0]
                else:
                    new_frame[i, j] = [0, 0, 0]  
        return new_frame

    def read_config(scenario_name: str) -> TestConfig:
        json_file = f"scenarios/{scenario_name}_cfg.json"
        with open(json_file, 'r') as f:
            raw_config = json.load(f)
        config = TestConfig()
        config.start = Waypoint(int(raw_config["start"][0]), int(raw_config["start"][1]), angle.new_deg(float(raw_config["start"][2])))
        config.goal = Waypoint(int(raw_config["goal"][0]), int(raw_config["goal"][1]), angle.new_deg(float(raw_config["goal"][2])))
        config.raw_frame = None
        
        img_file = f"scenarios/{scenario_name}.pfm"
        if os.path.exists(img_file):
            config.raw_frame = TestUtils.__convert_pfm(TestUtils.__read_pfm(img_file))
            config.file_type = PFM
        else:
            img_file = f"scenarios/{scenario_name}.png"
            if os.path.exists(img_file):
                config.raw_frame = np.array(cv2.imread(img_file))
            config.file_type = RGB
            
        config.segmentation_costs = raw_config["segmentation_costs"]
        config.segmentation_colors = raw_config["segmentation_colors"]
        config.lower_bound = raw_config["lower_bound"]
        config.upper_bound = raw_config["upper_bound"]
        config.og_real_size = raw_config["og_real_size"]
        config.max_curvature = raw_config["max_curvature"]
        config.max_steering_angle_deg = raw_config["max_steering_angle_deg"]
        config.vehicle_length_m = raw_config["vehicle_length_m"]
        config.meters_to_pixel_ratio = raw_config["meters_to_pixel_ratio"]
        config.pixel_to_meters_ratio = raw_config["pixel_to_meters_ratio"]

        world_origin_raw = raw_config["world_origin"]
        config.world_origin = WorldPose(
             lat=angle.new_deg(world_origin_raw[0]),
             lon=angle.new_deg(world_origin_raw[1]),
             alt=float(world_origin_raw[2]),
             compass=angle.new_deg(world_origin_raw[3])
        )
        return config

    def __draw_arrow(frame: np.ndarray, row: int, col: int, heading_deg: float, color=(0, 0, 255), thickness=2, length=20):
        rad = np.deg2rad(heading_deg)
        dx = int(np.cos(rad) * length)
        dy = int(-np.sin(rad) * length)
        start = (int(col), int(row))
        end = (int(col) + dx, int(row) + dy)
        cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.2)

    def export_color_frame(conf: TestConfig, file: str):
        f = SearchFrame(conf.raw_frame.shape[1], conf.raw_frame.shape[0], [-1,-1], [-1, -1])        
        f.set_frame_data(conf.raw_frame)
        f.set_class_costs(np.array(conf.segmentation_costs, dtype=np.float32))
        f.set_class_colors(np.array(conf.segmentation_colors, dtype=np.int32))
        color_frame = f.get_color_frame()
        TestUtils.__draw_arrow(color_frame, conf.start.z, conf.start.x,  90-conf.start.heading.deg())
        TestUtils.__draw_arrow(color_frame, conf.goal.z, conf.goal.x,  90-conf.goal.heading.deg(), color=(128, 30, 128))
        if file is None:
            return color_frame
        cv2.imwrite(file, color_frame)
        

    def export_planner_result(conf: TestConfig, file: str, path: list[Waypoint]):
        color_frame = TestUtils.export_color_frame(conf, None)
        for p in path:
            color_frame[p.z, p.x, :] = (255, 0, 0)
        # TestUtils.__draw_arrow(color_frame, conf.start.z, conf.start.x,  90-conf.start.heading.deg())
        # TestUtils.__draw_arrow(color_frame, conf.goal.z, conf.goal.x,  90-conf.goal.heading.deg(), color=(128, 30, 128))    
        cv2.imwrite(file, color_frame)

    def export_safe_distance_frame(frame: SearchFrame, file: str):
        outp = np.zeros((frame.height(), frame.width(), 3), dtype=np.uint8)
        for h in range (frame.height()):
            for w in range(frame.width()):
                outp[h, w, :] = [255, 255, 255]
                if frame.get_cost(w, h) < 0:
                    outp[h, w, :] = [0, 0, 255]
                else:
                    p = frame.get_traversability(w, h)
                    if p & 0xF00 > 0:
                        outp[h, w, :] = [0, 0, 0]
        cv2.imwrite(file, outp)

    def export_safe_distance_frame_minimal_dist_flag(frame: SearchFrame, file: str):
        outp = np.zeros((frame.height(), frame.width(), 3), dtype=np.uint8)
        for z in range (frame.height()):
            for x in range(frame.width()):
                outp[z, x, :] = [0, 0, 0]
                if not frame.is_obstacle(x, z):
                    outp[z, x, :] = [128, 128, 128]

                p = frame.get_traversability(x, z)
                if p & 0x100 > 0:
                    outp[z, x, :] = [255, 255, 255]

        cv2.imwrite(file, outp)


    def build_cuda_frame(conf: TestConfig) -> SearchFrame:
        f = SearchFrame(conf.raw_frame.shape[1], conf.raw_frame.shape[0], conf.lower_bound, conf.upper_bound)
        f.set_frame_data(conf.raw_frame)
        f.set_class_costs(np.array(conf.segmentation_costs, dtype=np.float32))
        f.set_class_colors(np.array(conf.segmentation_colors, dtype=np.int32))
        return f
        
    def build_cpu_frame(conf: TestConfig) -> SearchFrameCPU:
        f = SearchFrameCPU(conf.raw_frame.shape[1], conf.raw_frame.shape[0], conf.lower_bound, conf.upper_bound)
        f.set_frame_data(conf.raw_frame)
        f.set_class_costs(np.array(conf.segmentation_costs, dtype=np.float32))
        f.set_class_colors(np.array(conf.segmentation_colors, dtype=np.int32))
        return f
    
    def build_ego_params(conf: TestConfig) -> EgoParams:
        return EgoParams(
            search_frame_dimensions=(conf.raw_frame.shape[1], conf.raw_frame.shape[0]),
            search_frame_physical_dimensions=conf.og_real_size,
            ego_upper_bound=conf.upper_bound,
            ego_lower_bound=conf.lower_bound,
            max_curvature=conf.max_curvature,
            max_steering_angle=angle.new_deg(conf.max_steering_angle_deg),
            meters_to_pixel_ratio_width=conf.meters_to_pixel_ratio[0],
            meters_to_pixel_ratio_height=conf.meters_to_pixel_ratio[1],
            pixel_to_meters_ratio_width=conf.pixel_to_meters_ratio[0],
            pixel_to_meters_ratio_height=conf.pixel_to_meters_ratio[1],
            segmentation_class_colors=np.array(conf.segmentation_colors, dtype=np.int32),
            segmentation_class_costs=np.array(conf.segmentation_costs, dtype=np.float32),
            vehicle_length_m=conf.vehicle_length_m,
            world_origin=conf.world_origin
        )
    def build_search_params(conf: TestConfig, gpu: bool, timeout: int = 60000) -> SearchParams:
        frame = None
        if gpu:
            frame = TestUtils.build_cuda_frame(conf)
        else:
            frame = TestUtils.build_cpu_frame(conf)

        return SearchParams.init(
            start=conf.start,
            goal=conf.goal
            ).with_world_origin(conf.world_origin)\
            .with_distance_to_goal_tolerance(distance_px=15)\
            .with_velocity(velocity_m_s=1.0)\
            .with_map_origin(origin=MapPose(0, 0, 0, heading=angle.new_rad(0)))\
            .with_ego_pose(pose=MapPose(0, 0, 0, heading=conf.start.heading))\
            .with_heading_error_tolerance(angle.new_deg(5))\
            .with_timeout(timeout)\
            .with_max_path_size(40)\
            .with_frame(frame)\
            .build()
            

class TestTimer:
    # store start times for multiple keys
    start_time = {}

    @staticmethod
    def timed_exec(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{func.__name__}] {1000 * execution_time:.6f} ms")
        return result

    @staticmethod
    def exec_start(key: str = "default"):
        TestTimer.start_time[key] = time.time()

    @staticmethod
    def exec_stop(key: str = "default"):
        if key not in TestTimer.start_time:
            return None
        start = TestTimer.start_time.pop(key)
        end = time.time()
        execution_time = end - start
        print(f"[exec_{key}] {1000 * execution_time:.6f} ms")
        return execution_time


if __name__ == "__main__":
    conf = TestUtils.read_config("map_cost_5")
    TestUtils.export_color_frame(conf, "output.png")
    k = 1