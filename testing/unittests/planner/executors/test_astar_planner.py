import sys, time
sys.path.append("../../../../")
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import LocalPlanner, LocalPlannerType, PlanningResult, PlannerResultType
from data.coordinate_converter import CoordinateConverter
from testing.unittests.test_utils import *
import unittest
from utils.smoothness import Smoothness2D

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
        
    def __compute_cost(self, path: list[Waypoint]) -> float:
        sm = Smoothness2D()
        for p in path:
            sm.add_point(p.z, p.x)
        return sm.get_cost()
    
    def __run_plan(self, timeout: int, type: LocalPlannerType, planning_data: PlanningData, world_pose: WorldPose, outp: PlannerTestOutput) -> float:
        planner = LocalPlanner(
                plan_timeout_ms=timeout,
                local_planner_type=type,
                map_coordinate_converter=CoordinateConverter(world_pose)
            )
        
        planner.plan(planning_data)
        
        while planner.is_planning():
            time.sleep(0.01)
    
        res = planner.get_result()
        
        outp.add_path(res.path)
        outp.write("output_1.png")
        
        self.assertEqual(PlannerResultType.VALID, res.result_type)
        

        
        cost = self.__compute_cost(res.path)
        return cost

    def test_bev1(self):
        world_pose, planning_data, outp = self.read(1)
        timeout = 500
        
        cost = self.__run_plan(timeout, LocalPlannerType.AStar, planning_data, world_pose, outp)
        self.assertEqual(0.0, cost)
        
        cost = self.__run_plan(timeout, LocalPlannerType.HybridAStar, planning_data, world_pose, outp)
        self.assertEqual(0.0, cost)

        cost = self.__run_plan(timeout, LocalPlannerType.VectorialAStar, planning_data, world_pose, outp)
        self.assertEqual(0.0, cost)
        
        
        cost = self.__run_plan(timeout, LocalPlannerType.Overtaker, planning_data, world_pose, outp)
        self.assertEqual(0.0, cost)
        
        cost = self.__run_plan(timeout, LocalPlannerType.Interpolator, planning_data, world_pose, outp)
        self.assertEqual(0.0, cost)
    

if __name__ == "__main__":
    unittest.main()
