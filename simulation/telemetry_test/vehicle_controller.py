#
# This script intializes a Ego vehicle in Carla and attaches a Telemetry logger to direct-access vehicle sensor data
#

from carladriver import CarlaSimulation, CarlaEgoVehicle
from pydriveless import Telemetry
import time

ENABLE_TELEMETRY = True

def main():

    sim = CarlaSimulation(town_name="Town07")

    ego = sim.add_ego_vehicle(pos=(-70, 0, 3))
    gps = ego.attach_gps_sensor(period_ms=1000)
    imu = ego.attach_imu_sensor(period_ms=100)
    bev_rgb_camera = ego.init_rgb_bev_camera()

    Telemetry.setup(port=30000, enable=ENABLE_TELEMETRY)
    ego.set_carla_autopilot(True)

    while True:
        Telemetry.log(id="gps", data=gps.read())
        Telemetry.log(id="imu", data=imu.read())
        frame, _ = bev_rgb_camera.read()
        Telemetry.log(id="camera", data=frame)
        #print ("send data")
        time.sleep(0.5)


    pass


if __name__ == "__main__":
    main()