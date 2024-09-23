import sys, time
sys.path.append("../../")
sys.path.append("../../../")
import unittest, math
from model.world_pose import WorldPose
from model.map_pose import MapPose
from model.waypoint import Waypoint
from vision.occupancy_grid_cuda import OccupancyGrid, GridDirection
from data.coordinate_converter import CoordinateConverter
from planner.collision_detector import CollisionDetector
import numpy as np
import cv2
from model.physical_parameters import PhysicalParameters
from planner.planning_data_builder import PlanningDataBuilder, PlanningData
from slam.slam import SLAM
from model.sensors.gps import GPS
from model.sensors.imu import IMU
from model.sensors.odometer import Odometer
from model.sensor_data import GpsData, IMUData
from testing.test_utils import TestFrame


class TestPlanDataBuilder(PlanningDataBuilder):

    _frame: TestFrame
    
    def __init__(self, test_frame: TestFrame):
        self._frame = test_frame

    def build_planning_data(self) -> PlanningData:
        frame = self._frame.frame
        
        og = OccupancyGrid(
            frame=frame,
            minimal_distance_x=0,
            minimal_distance_z=0,
            lower_bound=Waypoint(1000,1000,0),
            upper_bound=Waypoint(1000,1000,0)
        )
        
        data = PlanningData(
            bev = frame,
            ego_location=MapPose(0, 0, 0, 0),
            goal=MapPose(40, 0, 0, 0),
            next_goal=MapPose(80, 0, 0, 0),
            velocity=10.0
        )

        data.og = og
        return data


class DummyGps(GPS):
    _data: GpsData
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set_from_str(self, raw_str: str) -> None:
        self._data = GpsData.from_str(raw_str)
    
    def read(self) -> GpsData:
        return self._data
    
class DummyIMU(GPS):
    _data: IMUData
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set_from_str(self, raw_str: str) -> None:
        self._data = IMUData.from_str(raw_str)
    
    def read(self) -> IMUData:
        return self._data
    
class DummyOdometer(Odometer):
    _data: float
    
    def __init__(self) -> None:
        self._data = None
        super().__init__()
    
    def set(self, val: float) -> None:
        self._data = val
    
    def read(self) -> float:
        return self._data


class TestCollisionDetector(unittest.TestCase):

    _collision: int
    
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self._collision = 0

    def on_collision_detected(self):
        self._collision += 1

    def test_collision_detect(self):
        
        self._collision = 0

        coord = CoordinateConverter(WorldPose(0, 0, 0, 0))
        gps = DummyGps()
        gps.set_from_str("11.1|11.2|1.2")
        
        imu = DummyIMU()
        imu.set_from_str("1.1|1.1|1.1|1.1|1.1|1.1|50")

        slam = SLAM(
            gps=gps,
            imu=imu,
            odometer=DummyOdometer()
        )
        
        slam.calibrate()
        
        frame = TestFrame(PhysicalParameters.OG_WIDTH, PhysicalParameters.OG_HEIGHT)       
        frame.add_obstacle(100, 100, 160, 120)
        builder = TestPlanDataBuilder(frame)
        

        cd = CollisionDetector(
            period_ms=1,
            coordinate_converter=coord,
            planning_data_builder=builder,
            slam=slam,
            on_collision_detected_cb=self.on_collision_detected
        )

       
        cd.start()
        
        path = [
            Waypoint(128, 128, 0),
            Waypoint(128, 108, 0),
            Waypoint(128, 88, 0),
            Waypoint(128, 68, 0),
            Waypoint(128, 48, 0),
            Waypoint(128, 28, 0),
            Waypoint(128, 0, 0),
        ]
        
        map_path = coord.convert_waypoint_path_to_map_pose(slam.estimate_ego_pose(), path)
        
        cd.watch_path(map_path)
        
        time.sleep(0.500)
        
        self.assertEqual(self._collision, 1)
        
        cd.destroy()


    def test_no_collision_detected(self):
        
        self._collision = False

        coord = CoordinateConverter(WorldPose(0, 0, 0, 0))
        gps = DummyGps()
        gps.set_from_str("11.1|11.2|1.2")
        
        imu = DummyIMU()
        imu.set_from_str("1.1|1.1|1.1|1.1|1.1|1.1|50")

        slam = SLAM(
            gps=gps,
            imu=imu,
            odometer=DummyOdometer()
        )
        
        slam.calibrate()
        
        frame = TestFrame(PhysicalParameters.OG_WIDTH, PhysicalParameters.OG_HEIGHT)       
        
        #mo obstacles
        #frame.add_obstacle(100, 100, 160, 120)
        builder = TestPlanDataBuilder(frame)
        

        cd = CollisionDetector(
            period_ms=1,
            coordinate_converter=coord,
            planning_data_builder=builder,
            slam=slam,
            on_collision_detected_cb=self.on_collision_detected
        )

       
        cd.start()
        
        path = [
            Waypoint(128, 128, 0),
            Waypoint(128, 108, 0),
            Waypoint(128, 88, 0),
            Waypoint(128, 68, 0),
            Waypoint(128, 48, 0),
            Waypoint(128, 28, 0),
            Waypoint(128, 0, 0),
        ]
        
        map_path = coord.convert_waypoint_path_to_map_pose(slam.estimate_ego_pose(), path)
        
        cd.watch_path(map_path)

        time.sleep(0.5)        
        
        self.assertEqual(self._collision, 0)
        
        cd.destroy()

        
if __name__ == "__main__":
    unittest.main()

        