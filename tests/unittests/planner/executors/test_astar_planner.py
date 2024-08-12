import sys, time
sys.path.append("../../../../")
import unittest
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType, PlanningResult, PlannerResultType
from model.planning_data import PlanningData
import cv2, numpy as np, json
from vision.occupancy_grid_cuda import SEGMENTED_COLORS, OccupancyGrid
from model.map_pose import MapPose
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from data.coordinate_converter import CoordinateConverter
from planner.goal_point_discover import GoalPointDiscover
from model.physical_parameters import PhysicalParameters


class DummyLocalPlanner (LocalPathPlannerExecutor):
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
    
    def check_timeout(self) -> bool:
        return self._check_timeout()


class PlannerDataReader:

    @classmethod
    def __convert_frame(cls, colored_f: np.ndarray) -> np.ndarray:
        original_f = np.zeros(colored_f.shape, dtype=np.uint8)
        for i in range (0, colored_f.shape[0]):
            for j in range (0, colored_f.shape[1]):
                for k in range (0, len(SEGMENTED_COLORS)):
                    if colored_f[i, j, 0] == SEGMENTED_COLORS[k][0] and\
                        colored_f[i, j, 1] == SEGMENTED_COLORS[k][1] and\
                        colored_f[i, j, 2] == SEGMENTED_COLORS[k][2]:
                        original_f[i, j, 0] = k
                        break
        return original_f      

    @classmethod
    def read (cls, log_num: int) -> tuple[WorldPose, PlanningData]:
        bev = cls.__convert_frame(cv2.imread(f"imgs/bev_{log_num}.png"))
        with open(f"imgs/log_{log_num}.log") as f:
            
            lines = f.readlines()
            log_data = json.loads(lines[0])
            pos_log_data = json.loads(lines[1])
            
            world_pose = WorldPose.from_str(log_data["gps"])
            
            return world_pose, PlanningData(
                bev,
                ego_location=MapPose.from_str(log_data["pose"]),
                velocity=float(log_data["velocity"]),
                goal=MapPose.from_str(pos_log_data["goal"]),
                next_goal=MapPose.from_str(pos_log_data["next_goal"])
            )
            
            
            
            

class TestAStarPlanner(unittest.TestCase):
    
    def test_a_start(self):
        
        world_pose, planning_data = PlannerDataReader.read(1)
        
        planner = LocalPlanner(
                plan_timeout_ms=500,
                local_planner_type=LocalPlannerType.AStar,
                map_coordinate_converter=CoordinateConverter(world_pose)
            )
        
        self.assertFalse(planner.is_planning())
        self.assertIsNone(planner.get_result())
        
        planner.plan(planning_data)
        time.sleep(0.1)
        self.assertTrue(planner.is_planning())
        
        while planner.is_planning():
            time.sleep(0.01)
            
        self.assertFalse(planner.is_planning())
        res = planner.get_result()
        
        self.assertEqual(PlannerResultType.VALID, res.result_type)
            
      
    

if __name__ == "__main__":
    unittest.main()
