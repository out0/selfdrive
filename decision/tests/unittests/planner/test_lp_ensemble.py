import sys, time
sys.path.append("../../../")
from pydriveless import MapPose, Waypoint, WorldPose, angle
from pydriveless import SearchFrame
from pydriveless import CoordinateConverter
import unittest
import numpy as np
from ensemble.planner.ensemble import Ensemble
from ensemble.model.planning_data import PlanningData
from ensemble.model.planning_result import PlanningResult, PlannerResultType
from ensemble.model.physical_paramaters import PhysicalParameters
import cv2
import cProfile, timeit

class TestLPEnsemble(unittest.TestCase):

    TIMEOUT_MS = 1000
    ORIGIN = WorldPose(angle.new_rad(0), angle.new_rad(0), 0, angle.new_rad(0))

    __last_planner: any
    __last_planning_data: any

    @classmethod
    def run_planner(cls) -> None:
        cls.__last_planner.plan(cls.__last_planning_data, run_in_main_thread=True)


    def test_free_area(self):
        return
        bev = np.full((100, 100, 3), fill_value=0.0, dtype=np.float32)
        og = SearchFrame(width=100, height=100, lower_bound=(-1, -1), upper_bound=(-1, -1))
        og.set_frame_data(bev)
        og.set_class_costs(np.array([0.0, -1.0]))
        
        conv = CoordinateConverter(origin=TestLPHybridA.ORIGIN, width=100, height=100, perceptionHeightSize_m=1, perceptionWidthSize_m=1)
        planner = HybridAStar(conv, max_exec_time_ms=TestLPHybridA.TIMEOUT_MS)
        
        ego_location = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        g1 = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        L2 = Waypoint(x=50, z=-100, heading=angle.new_rad(0))
        g2: MapPose = conv.convert(ego_location, L2)
        
        planning_data = PlanningData(
            og=og,
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(5, 5)
        )
        
        planning_data.set_local_goal(Waypoint(x=50, z=0, heading=angle.new_deg(0.0)))
        og.process_distance_to_goal(planning_data.local_goal().x, planning_data.local_goal().z)
        
        TestLPHybridA.__last_planning_data = planning_data
        TestLPHybridA.__last_planner = planner

        cProfile.run("TestLPHybridA.run_planner()")
        
        # planner.plan(planning_data)
        # while planner.is_planning():
        #     pass


        
        self.assertTrue(planner.get_execution_time() > 0)
        
        result = planner.get_result()
        
        self.assertEqual(result.result_type, PlannerResultType.VALID)
        
        for p in result.path:
            if (p.x > 52 or p.x < 48):
                self.fail("should be straight or near straight line")
        
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))
        f = og.get_color_frame()
        for p in result.path:
            f[p.z, p.x] = [0, 255, 0]
                
        cv2.imwrite("debug.png", f)
       
    def test_diverge_plan_due_to_obstacle(self):
        return
        bev = np.full((100, 400, 3), fill_value=0.0, dtype=np.float32)
        og = SearchFrame(width=100, height=400, lower_bound=(-1, -1), upper_bound=(-1, -1))
        
        for z in range(0, 10):
            for x in range(40, 60):
                bev[z,x,0] = 1.0
        
        og.set_frame_data(bev)
        og.set_class_costs(np.array([0.0, -1.0]))
        og.set_class_colors(np.array([(0, 0, 0), (255, 255, 255)]))

        f = og.get_color_frame()
        cv2.imwrite("debug.png", f)
        
        conv = CoordinateConverter(origin=TestLPHybridA.ORIGIN, width=100, height=400, perceptionHeightSize_m=4, perceptionWidthSize_m=1)
        planner = HybridAStar(conv, max_exec_time_ms=TestLPHybridA.TIMEOUT_MS)
        
        ego_location = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        g1 = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        L2 = Waypoint(x=50, z=-100, heading=angle.new_rad(0))
        g2: MapPose = conv.convert(ego_location, L2)
        
        planning_data = PlanningData(
            og=og,
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(5, 5)
        )
        
        planning_data.set_local_goal(Waypoint(x=15, z=0, heading=angle.new_deg(0.0)))

        execution_time = timeit.timeit(lambda: og.process_distance_to_goal(planning_data.local_goal().x, planning_data.local_goal().z), number=1)
        print(f"Execution time: {execution_time:.6f} seconds")

        # planner.plan(planning_data)
        # while planner.is_planning():
        #     pass
        TestLPHybridA.__last_planning_data = planning_data
        TestLPHybridA.__last_planner = planner

        cProfile.run("TestLPHybridA.run_planner()")

        result = planner.get_result()

        #self.assertEqual(result.result_type, PlannerResultType.VALID)
        planner.cancel()
        while planner.is_running():
            pass

        print(str(result))

        f = og.get_color_frame()
        if result.path is not None:
            for p in result.path:
                f[p.z, p.x] = [0, 255, 0]
        
        cv2.imwrite("debug.png", f)
        p = 1


    def test_bev_1(self):
        bev = np.array(cv2.imread("bev_1.png"), dtype=np.float32)
        og = SearchFrame(width=bev.shape[1], height=bev.shape[0], lower_bound=PhysicalParameters.EGO_LOWER_BOUND, upper_bound=PhysicalParameters.EGO_UPPER_BOUND)
        
        og.set_frame_data(bev)
        og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        og.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)

        f = og.get_color_frame()
        cv2.imwrite("debug.png", f)
        
        conv = CoordinateConverter(origin=TestLPEnsemble.ORIGIN, width=PhysicalParameters.OG_WIDTH, height=PhysicalParameters.OG_HEIGHT, perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT, perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH)
        planner = Ensemble(conv, TestLPEnsemble.TIMEOUT_MS)
        
        ego_location = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        g1 = MapPose(x=0, y=0, z=0, heading=angle.new_rad(0.0))
        L2 = Waypoint(x=50, z=-100, heading=angle.new_rad(0))
        g2: MapPose = conv.convert(ego_location, L2)
        
        planning_data = PlanningData(
            seq=0,
            og=og,
            start=Waypoint(128,128, heading=angle.new_rad(0)),
            ego_location=ego_location,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(5, 5)
        )
        
        planning_data.set_local_goal(Waypoint(x=108, z=0, heading=angle.new_deg(0.0)))

        execution_time = timeit.timeit(lambda: og.process_distance_to_goal(planning_data.local_goal().x, planning_data.local_goal().z), number=1)
        print(f"Execution time: {execution_time:.6f} seconds")

        planner.plan(planning_data)
        while planner.is_planning():
            if planner.new_path_available():
                result = planner.get_result()
                print("\nnew path")
                print(str(result))
            pass

        #result = planner.get_result()

        #self.assertEqual(result.result_type, PlannerResultType.VALID)
        planner.cancel()
        while planner.is_running():
            pass

        #print(str(result))

        f = og.get_color_frame()
        if result.path is not None:
            for p in result.path:
                f[p.z, p.x] = [0, 255, 0]
        
        cv2.imwrite("debug.png", f)
        p = 1


if __name__ == "__main__":
    unittest.main()
