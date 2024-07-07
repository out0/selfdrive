import sys, time
sys.path.append("../../../")
sys.path.append("../../")
import unittest, math

from model.world_pose import WorldPose
from model.map_pose import MapPose
from data.coordinate_converter import CoordinateConverter


class TestCoordinateConverter(unittest.TestCase):

    def __test_conversion_to_map(self, conv: CoordinateConverter, pose: str, map_truth: str):
        world_pose = WorldPose.from_str(pose)
        map_pose = conv.convert_to_map_pose(world_pose)
        map_pose_carla = MapPose.from_str(map_truth)
        self.assertAlmostEqual(map_pose.x, map_pose_carla.x, 1)
        self.assertAlmostEqual(map_pose.y, map_pose_carla.y, 1)
        self.assertAlmostEqual(map_pose.heading, map_pose_carla.heading, 0)
        
    def __test_conversion_to_world(self, conv: CoordinateConverter, pose: str, map_truth: str):
        world_pose_truth = WorldPose.from_str(pose)
        map_pose = MapPose.from_str(map_truth)
        world_pose = conv.convert_to_world_pose(map_pose)
        
        self.assertAlmostEqual(world_pose_truth.lat, world_pose.lat, 7)
        self.assertAlmostEqual(world_pose_truth.lon, world_pose.lon, 7)
        self.assertAlmostEqual(world_pose_truth.heading, world_pose.heading, 0)

    def test_convert_world_pose_to_map_pose(self):
        origin = WorldPose.from_str("-4.256008878655848e-09|-1.5864868596990013e-08|1.0150023698806763|89.99923752329113")
        conv = CoordinateConverter(origin)
        self.__test_conversion_to_map(conv, "6.008207265040255e-10|-9.445475428916048e-09|1.0149040222167969|90.01170944871556", "-0.0010514655150473118|-6.688445864710957e-05|1.0149040222167969|0.011707369238138199")
        self.__test_conversion_to_map(conv, "-8.14085865386005e-09|-1.782302467980099e-08|1.0143463611602783|359.9802297900763", "-0.0019840500317513943|0.0009062352473847568|1.0143463611602783|-90.00768280029297")
        self.__test_conversion_to_map(conv, "-1.3355261785363837e-08|-2.011051673862556e-08|1.013861060142517|314.99925402977027", "-0.002238692482933402|0.0014867002610117197|1.013861060142517|-135.0007781982422")
        self.__test_conversion_to_map(conv, "3.5947697853089267e-05|-3.592238022246919e-05|1.0108108520507812|315.0036526715958", "-3.998861074447632|-4.001679420471191|1.0108108520507812|-134.99636840820312")
        self.__test_conversion_to_map(conv, "0.00017966245712841555|-3.5935588398026246e-05|1.0016272068023682|0.0", "-4.000331401824951|-19.99993324279785|1.0016272068023682|-89.99999237060547")
        self.__test_conversion_to_map(conv, "-0.00017966285120962766|-3.593243145773581e-05|1.0016334056854248|180.00000500895632", "-3.9999799728393555|19.999977111816406|1.0016334056854248|90.00001525878906")
        self.__test_conversion_to_map(conv, "-0.0005389866003611132|4.491557144842781e-05|1.0365043878555298|90.00011178750489", "4.999978542327881|59.99971389770508|1.0365043878555298|0.00010962043597828597")
        self.__test_conversion_to_map(conv, "-0.0005389890676639197|4.4915811324487864e-05|1.036553144454956|180.01979889717103", "5.00000524520874|59.9999885559082|1.036553144454956|90.00000762939453")
        self.__test_conversion_to_map(conv, "-0.000538989855826344|4.491587986050503e-05|1.0365053415298462|225.00000967629", "5.0000128746032715|60.00007629394531|1.0365053415298462|135.0")
 
    def test_convert_map_pose_to_world_pose(self):
        origin = WorldPose.from_str("-4.256008878655848e-09|-1.5864868596990013e-08|1.0150023698806763|89.99923752329113")
        conv = CoordinateConverter(origin)
        self.__test_conversion_to_world(conv, "6.008207265040255e-10|-9.445475428916048e-09|1.0149040222167969|90.01170944871556", "-0.0010514655150473118|-6.688445864710957e-05|1.0149040222167969|0.011707369238138199")
        self.__test_conversion_to_world(conv, "-8.14085865386005e-09|-1.782302467980099e-08|1.0143463611602783|359.9802297900763", "-0.0019840500317513943|0.0009062352473847568|1.0143463611602783|-90.00768280029297")
        self.__test_conversion_to_world(conv, "-1.3355261785363837e-08|-2.011051673862556e-08|1.013861060142517|314.99925402977027", "-0.002238692482933402|0.0014867002610117197|1.013861060142517|-135.0007781982422")
        self.__test_conversion_to_world(conv, "3.5947697853089267e-05|-3.592238022246919e-05|1.0108108520507812|315.0036526715958", "-3.998861074447632|-4.001679420471191|1.0108108520507812|-134.99636840820312")
        self.__test_conversion_to_world(conv, "0.00017966245712841555|-3.5935588398026246e-05|1.0016272068023682|0.0", "-4.000331401824951|-19.99993324279785|1.0016272068023682|-89.99999237060547")
        self.__test_conversion_to_world(conv, "-0.00017966285120962766|-3.593243145773581e-05|1.0016334056854248|180.00000500895632", "-3.9999799728393555|19.999977111816406|1.0016334056854248|90.00001525878906")
        self.__test_conversion_to_world(conv, "-0.0005389866003611132|4.491557144842781e-05|1.0365043878555298|90.00011178750489", "4.999978542327881|59.99971389770508|1.0365043878555298|0.00010962043597828597")
        self.__test_conversion_to_world(conv, "-0.0005389890676639197|4.4915811324487864e-05|1.036553144454956|180.01979889717103", "5.00000524520874|59.9999885559082|1.036553144454956|90.00000762939453")
        self.__test_conversion_to_world(conv, "-0.000538989855826344|4.491587986050503e-05|1.0365053415298462|225.00000967629", "5.0000128746032715|60.00007629394531|1.0365053415298462|135.0")


if __name__ == "__main__":
    unittest.main()



#
# Data built with the following script
#
# ---------------------------------------------

# import sys, os
# sys.path.append("../../")
# from carlasim.carla_client import CarlaClient
# from carlasim.carla_ego_car import CarlaEgoCar
# from carlasim.sensors.data_sensors import *

# client = CarlaClient(town='Town07')
# ego = CarlaEgoCar(client)
# ego.init_fake_bev_seg_camera()
# ego.set_pose(0, 0, 0, 0)
# ego.set_brake(1.0)

# import time
# from model.map_pose import MapPose
# from data.coordinate_converter import CoordinateConverter
# from model.world_pose import WorldPose

# ego.set_pose(0, 0, 0, 0)
# time.sleep(1)
# ego.set_brake(1.0)
# d = ego.get_gps().read()

# initial_world_pose = WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)
# p = ego.get_gps().get_location()
# map_carla = MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2))

# print(f"GPS: {d.latitude}, {d.longitude}, {d.altitude}")
# print(f"Base World Pose: {initial_world_pose}")
# print(f"Location in Carla: {map_carla}")

# conv = CoordinateConverter(initial_world_pose)
# map_pose1 = conv.convert_to_map_pose(initial_world_pose)
# print(f"Initial map pose: {map_pose1}")

# ego.set_pose(-3, -22, 1, 90)
# time.sleep(1)
# ego.set_brake(1.0)
# d = ego.get_gps().read()
# time.sleep(1)
# wp2 = WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)
# p = ego.get_gps().get_location()
# map_truth = MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2))
# print(f"GPS: {d.latitude}, {d.longitude}, {d.altitude}")
# print(f"World Pose: {wp2}")
# print(f"Location in Carla: {map_truth}")
# converted_map = conv.convert_to_map_pose(wp2)
# print(converted_map)
# print(f"Converted map: {converted_map}")

# poses = [
#     (0, 0, 1, 0),
#     (0, 0, 1, -90),
#     (0, 0, 1, -135),
#     (-4, -4, 1, -135),
#     (-4, -20, 1, -90),
#     (-4, 20, 1, 90),
#     (5, 60, 1, 0),
#     (5, 60, 1, 90),
#     (5, 60, 1, 135),
# ]

# for i in range (len(poses)):
#     ego.set_pose(poses[i][0], poses[i][1], poses[i][2], poses[i][3])
#     time.sleep(3)
#     d = ego.get_gps().read()
#     time.sleep(1)
#     print(WorldPose(lat=d.latitude, lon=d.longitude, alt=d.altitude, heading=ego.get_imu().read().compass)) 
#     p = ego.get_gps().get_location()
#     print(MapPose(round(p[0],2), round(p[1], 2), round(p[2], 2), round(ego.get_heading(), 2)))
    
    