import sys, time
sys.path.append("../../../../")
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from testing.unittests.test_utils import *
import unittest

class DummyLocalPlanner (LocalPathPlannerExecutor):
    def __init__(self, max_exec_time_ms: int) -> None:
        super().__init__(max_exec_time_ms)
    
    def check_timeout(self) -> bool:
        return self._check_timeout()

           

class TestLocalPlanners(unittest.TestCase):
    _last_world_pose: WorldPose
    _last_planning_data: PlanningData
    _last_test_outp: PlannerTestOutput
    _last_code: int
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._last_code = -1
    
    def read(self, code: int) -> tuple[WorldPose, PlanningData, PlannerTestOutput]:
        
        if self._last_code != code:
            self._last_world_pose, self._last_planning_data = PlannerDataReader.read(1)
            self._last_test_outp = PlannerTestOutput(self._last_planning_data.og.get_color_frame())
            
        return self._last_world_pose, self._last_planning_data, self._last_test_outp
    
    def run_planner(self, code: int, planner_type: LocalPlannerType, color: list, timeout = 500):
        
        world_pose, planning_data, outp = self.read(code)
        
        planner = LocalPlanner(
                plan_timeout_ms=timeout,
                local_planner_type=planner_type,
                map_coordinate_converter=CoordinateConverter(world_pose)
            )
        
        self.assertFalse(planner.is_planning())
        self.assertIsNone(planner.get_result())
        
        planner.plan(planning_data)
        
        while planner.is_planning():
            time.sleep(0.01)
            
        self.assertFalse(planner.is_planning())
        res = planner.get_result()
        
        self.assertEqual(PlannerResultType.VALID, res.result_type)
            
        outp.add_path(res.path, color=color)

    def test_bev1(self):
        world_pose, planning_data, outp = self.read(1)
        timeout = 500
        
        planner = LocalPlanner(
                plan_timeout_ms=timeout,
                local_planner_type=LocalPlannerType.AStar,
                map_coordinate_converter=CoordinateConverter(world_pose)
            )
        
        
    

if __name__ == "__main__":
    unittest.main()
