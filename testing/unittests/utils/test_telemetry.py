import sys
sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
from model.waypoint import Waypoint
import numpy as np
from utils.cudac.cuda_frame import *
from utils.telemetry import Telemetry
from model.planning_data import *
import time, os
                
class TestTelemetry(unittest.TestCase): 
    
    def test_simple_log(self):
    
        os.system("rm -rf planning_data/")
        Telemetry.initialize()
        
        pre_planning = PlanningData(
            bev = np.full((20,20,3), 5),
            ego_location=MapPose(
                101.1,
                101.2,
                101.3,
                101.4
            ),
            goal=None,
            next_goal=None,
            velocity=100.12
        )

        Telemetry.log_pre_planning_data(pre_planning)
        
        planning = PlanningData(
            bev = np.full((20,20,3), 5),
            ego_location=MapPose(
                102.1,
                102.2,
                102.3,
                heading=102.4
            ),
            goal=MapPose(
                22.1,
                22.2,
                22.3,
                22.14
            ),
            next_goal=MapPose(
                33.1,
                33.2,
                33.3,
                33.14
            ),
            velocity=99.12
        )
        
        res = PlanningResult(
            planner_name='unittesting',
            ego_location=MapPose(
                103.1,
                103.2,
                103.3,
                heading=103.4
            ),
            goal=MapPose(
                222.1,
                222.2,
                222.3,
                222.14
            ),
            next_goal=MapPose(
                333.1,
                333.2,
                333.3,
                333.14
            ),
            direction=12,
            local_goal=Waypoint(0, 1, 2.2),
            local_start=Waypoint(1, 0, 3.3),
            path=[
                Waypoint(0, 1, 2.2),
                Waypoint(1, 1, 2.2),
                Waypoint(2, 1, 2.2),
                Waypoint(3, 1, 2.2),
            ],
            result_type=PlannerResultType.INVALID_PATH,
            timeout=True,
            total_exec_time_ms=12.32
        )
        
        Telemetry.log_planning_data(planning, res)
        
        time.sleep(0.5)
        
        bev = Telemetry.read_planning_bev(1)
        stored_res = Telemetry.read_planning_result(1)
        p_res = Telemetry.read_pre_planning_result(1)
        
        for i in range (bev.shape[0]):
            for j in range (bev.shape[1]):
                for k in range (bev.shape[2]):
                    self.assertEqual(bev[i,j,k], planning.bev[i,j,k])                    
        
        self.assertEqual(res.result_type, stored_res.result_type)
        self.assertEqual(res.path, stored_res.path)
        self.assertEqual(res.timeout, stored_res.timeout)
        self.assertEqual(res.planner_name, stored_res.planner_name)
        self.assertEqual(res.total_exec_time_ms, stored_res.total_exec_time_ms)
        self.assertEqual(res.local_start, stored_res.local_start)
        self.assertEqual(res.local_goal, stored_res.local_goal)
        self.assertEqual(res.goal_direction , stored_res.goal_direction)
        self.assertEqual(res.ego_location , stored_res.ego_location)
        self.assertEqual(res.map_goal, stored_res.map_goal)
        self.assertEqual(res.map_next_goal, stored_res.map_next_goal)
        self.assertEqual(res.total_exec_time_ms, stored_res.total_exec_time_ms)


        self.assertEqual(p_res.result_type, PlannerResultType.NONE)
        self.assertEqual(p_res.path, [])
        self.assertEqual(p_res.timeout, False)
        self.assertEqual(p_res.planner_name, "-")
        self.assertEqual(p_res.total_exec_time_ms, 0)
        self.assertEqual(p_res.local_start, None)
        self.assertEqual(p_res.local_goal, None)
        self.assertEqual(p_res.goal_direction , 0)
        self.assertEqual(p_res.ego_location , pre_planning.ego_location)
        self.assertEqual(p_res.map_goal, None)
        self.assertEqual(p_res.map_next_goal, None)

  
    def test_many_logs(self):
        
        os.system("rm -rf planning_data/")
        Telemetry.initialize()
        
        pre_planning = PlanningData(
            bev = np.full((20,20,3), 5),
            ego_location=MapPose(
                101.1,
                101.2,
                101.3,
                101.4
            ),
            goal=None,
            next_goal=None,
            velocity=100.12
        )

        planning = PlanningData(
            bev = np.full((20,20,3), 5),
            ego_location=MapPose(
                102.1,
                102.2,
                102.3,
                heading=102.4
            ),
            goal=MapPose(
                22.1,
                22.2,
                22.3,
                22.14
            ),
            next_goal=MapPose(
                33.1,
                33.2,
                33.3,
                33.14
            ),
            velocity=99.12
        )
        
        res = PlanningResult(
            planner_name='unittesting',
            ego_location=MapPose(
                103.1,
                103.2,
                103.3,
                heading=103.4
            ),
            goal=MapPose(
                222.1,
                222.2,
                222.3,
                222.14
            ),
            next_goal=MapPose(
                333.1,
                333.2,
                333.3,
                333.14
            ),
            direction=12,
            local_goal=Waypoint(0, 1, 2.2),
            local_start=Waypoint(1, 0, 3.3),
            path=[
                Waypoint(0, 1, 2.2),
                Waypoint(1, 1, 2.2),
                Waypoint(2, 1, 2.2),
                Waypoint(3, 1, 2.2),
            ],
            result_type=PlannerResultType.INVALID_PATH,
            timeout=True,
            total_exec_time_ms=12.32
        )
        
        for _ in range (0, 10):
            Telemetry.log_pre_planning_data(pre_planning)
            Telemetry.log_planning_data(planning, res)
        
        time.sleep(1)
        
        for i in range (1, 11):
            self.assertTrue(os.path.exists(f"planning_data/bev_{i}.png"))
            self.assertTrue(os.path.exists(f"planning_data/pre_bev_{i}.png"))
            self.assertTrue(os.path.exists(f"planning_data/planning_result_{i}.json"))
        
        
        
        bev = Telemetry.read_planning_bev(1)
        stored_res = Telemetry.read_planning_result(1)
        p_res = Telemetry.read_pre_planning_result(1)
        
        for i in range (bev.shape[0]):
            for j in range (bev.shape[1]):
                for k in range (bev.shape[2]):
                    self.assertEqual(bev[i,j,k], planning.bev[i,j,k])                    
        
        self.assertEqual(res.result_type, stored_res.result_type)
        self.assertEqual(res.path, stored_res.path)
        self.assertEqual(res.timeout, stored_res.timeout)
        self.assertEqual(res.planner_name, stored_res.planner_name)
        self.assertEqual(res.total_exec_time_ms, stored_res.total_exec_time_ms)
        self.assertEqual(res.local_start, stored_res.local_start)
        self.assertEqual(res.local_goal, stored_res.local_goal)
        self.assertEqual(res.goal_direction , stored_res.goal_direction)
        self.assertEqual(res.ego_location , stored_res.ego_location)
        self.assertEqual(res.map_goal, stored_res.map_goal)
        self.assertEqual(res.map_next_goal, stored_res.map_next_goal)
        self.assertEqual(res.total_exec_time_ms, stored_res.total_exec_time_ms)


        self.assertEqual(p_res.result_type, PlannerResultType.NONE)
        self.assertEqual(p_res.path, [])
        self.assertEqual(p_res.timeout, False)
        self.assertEqual(p_res.planner_name, "-")
        self.assertEqual(p_res.total_exec_time_ms, 0)
        self.assertEqual(p_res.local_start, None)
        self.assertEqual(p_res.local_goal, None)
        self.assertEqual(p_res.goal_direction , 0)
        self.assertEqual(p_res.ego_location , pre_planning.ego_location)
        self.assertEqual(p_res.map_goal, None)
        self.assertEqual(p_res.map_next_goal, None)




if __name__ == "__main__":
    unittest.main()