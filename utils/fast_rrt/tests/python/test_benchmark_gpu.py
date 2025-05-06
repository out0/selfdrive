import sys, os
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/selfdrive")
import unittest
from utils.fast_rrt.fastrrt import FastRRT
from test_utils import TestFrame, TestData, TestUtils
import time, math
from curve_quality import CurveAssessment
import cv2

MAX_STEERING_ANGLE = 40
VEHICLE_LENGTH_M = 5.412658774
TIMEOUT = 60000

def get_test_data(file: str) -> TestData:
    return TestFrame(file).get_data_cuda()

class TestScenario:
    file: str
    custom_start: tuple[int, int, float]
    custom_goal: tuple[int, int, float]
    custom_start_heading: float
    custom_goal_heading: float

    def __init__(self, 
                 file: str, 
                 custom_start_heading: float = None,
                 custom_goal_heading: float = None,
                 custom_start: tuple[int, int, float] = None,
                 custom_goal: tuple[int, int, float] = None):
        self.file = file
        self.custom_start = custom_start
        self.custom_goal = custom_goal
        self.custom_start_heading = custom_start_heading
        self.custom_goal_heading = custom_goal_heading  


class TestFastRRT(unittest.TestCase):
    
    def execute_scenario(self, scenario: TestScenario, smart: bool = True, path_step_size: float = 50.0, dist_to_goal_tolerance: float = 15.0, optim_loop_count: int = 20):
        
        print (f"executing scenario {scenario.file}")
        
        data = TestUtils.timed_exec(get_test_data, f"test_scenarios/{scenario.file}.png")

        rrt = FastRRT(
            width=data.width(),
            height=data.height(),
            perception_height_m=data.real_height(),
            perception_width_m=data.real_width(),
            max_steering_angle_deg=MAX_STEERING_ANGLE,
            vehicle_length_m=VEHICLE_LENGTH_M,
            timeout_ms=TIMEOUT,
            min_dist_x=22,
            min_dist_z=40,
            lower_bound_x=-1,
            lower_bound_z=-1,
            upper_bound_x=-1,
            upper_bound_z=-1,
            max_path_size_px=path_step_size,
            dist_to_goal_tolerance_px=dist_to_goal_tolerance,
            libdir=None
        )
        
        start = scenario.custom_start if scenario.custom_start else (data.start.x, data.start.z, 0.0)
        goal = scenario.custom_goal if scenario.custom_goal else (data.goal.x, data.goal.z, 0.0)
        
        if scenario.custom_start_heading is not None:
            start = (start[0], start[1], scenario.custom_start_heading)
        if scenario.custom_goal_heading is not None:
            goal = (goal[0], goal[1], scenario.custom_goal_heading)
        
        rrt.set_plan_data(
            data.frame.get_cuda_frame(),
            start=start,
            goal=goal,
            velocity_m_s=1.0
        )
        
        
        start_time = time.time()
        rrt.search_init()
        loop_count = 0
        while not rrt.goal_reached() and rrt.loop(smart):
            loop_count += 1       
        end_time = time.time()
        execution_time = end_time - start_time
        
        path = rrt.get_planned_path(interpolate=True)
        if path is None:
            print(f"[{scenario.file}] no path found")
            return False

        coarse_data = CurveAssessment(data.width(), data.height()).assess_curve(path, start[2])
        coarse_data.num_loops = loop_count
        coarse_data.proc_time_ms = execution_time * 1000
        coarse_data.timeout = TIMEOUT > 0 and coarse_data.proc_time_ms >= TIMEOUT
        coarse_data.goal_reached = rrt.goal_reached()
        coarse_data.coarse = True
        coarse_data.name = f"gpu_{scenario.file}"
        coarse_data.curve = rrt.get_planned_path(interpolate=False)
  
        print(f"[{scenario.file}] coarse path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
        TestUtils.output_path_result(data.frame, path, f"test_result/coarse_{coarse_data.name}.png")       
        
       
        start_time = time.time()
        loop_count = 0
        while loop_count < optim_loop_count and rrt.loop_optimize():
            loop_count += 1
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"[{scenario.file}] optim path total: {1000 * execution_time:.6f} ms, mean: {1000 * (execution_time/loop_count):.6f} ms/loop, num loops: {loop_count}")
        
        path = rrt.get_planned_path(interpolate=True)
        
        optim_data = CurveAssessment(data.width(), data.height()).assess_curve(path, start[2])
        optim_data.num_loops = loop_count
        optim_data.proc_time_ms = execution_time * 1000
        optim_data.timeout = TIMEOUT > 0 and optim_data.proc_time_ms >= TIMEOUT
        optim_data.goal_reached = rrt.goal_reached()
        optim_data.coarse = False
        optim_data.name = f"gpu_{scenario.file}"
        
        TestUtils.output_path_result(data.frame, path, f"test_result/optim_{coarse_data.name}.png")
        optim_data.curve = rrt.get_planned_path(interpolate=False)
        
        result_file = f"test_result/results.csv"
        data_result_file = f"test_result/data_results.csv"
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write(coarse_data.to_csv_header())
                
        with open(result_file, "a") as f:
            f.write(coarse_data.to_csv())
            f.write(optim_data.to_csv())
            
        with open(data_result_file, "a") as f:
            f.write(coarse_data.to_json())
            f.write(optim_data.to_json())
        

    def test_gpu_scenarios(self):

        self.execute_scenario(TestScenario("large_1"))

        self.execute_scenario(TestScenario("large_2",
                                           custom_start_heading=math.radians(90), 
                                           custom_goal_heading=math.radians(45)))

        self.execute_scenario(TestScenario("large_3",
                                           custom_start_heading=math.radians(90), 
                                           custom_goal_heading=math.radians(45)))

        self.execute_scenario(TestScenario("small_1"), smart=False)

        self.execute_scenario(TestScenario("small_2"), smart=False)
        
        self.execute_scenario(TestScenario("small_3"), smart=False)        


if __name__ == "__main__":
    unittest.main()