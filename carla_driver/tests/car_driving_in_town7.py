import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) + "../../")))
#sys.path.append("/home/cristiano/Documents/Projects/Mestrado/code/driveless-new/carla_driver")
#sys.path.append(os.getcwd())
#sys.path.append(os.getcwd() +"/libdriveless")

from pydriveless import EgoVehicle
from carladriver import CarlaSimulation, CarlaEgoVehicle
import threading, time


def sensor_show(sim: CarlaSimulation, ego_vehicle: EgoVehicle):
    
    before = time.time()
    while True:
        if time.time() - before >= 1:
            # print(f"[Vehicle Speed] {ego_vehicle.get_speed_vehicle()} m/s")
            # print(f"[Vehicle Velocity] {ego_vehicle.get_velocity_vehicle()}")
            # print(f"[Vehicle Angular Velocity] {ego_vehicle.get_angular_velocity_vehicle()}")
            # print(f"[Vehicle Pose] {ego_vehicle.get_pose_vehicle()}")

            gps_data = ego_vehicle.read_gps()
            print (f"[GPS] {gps_data}")

            imu_data = ego_vehicle.read_imu()
            print (f"[IMU] {imu_data}")
        
            print("[Vehicle Acceleration] ", ego_vehicle.get_accel_vehicle())
            
            ax, ay, az = ego_vehicle.get_accel_vehicle()
            print(f"[Accel ratio] {imu_data.accel_x / ax if ax != 0 else 0}, {imu_data.accel_y / ay if ay != 0 else 0}, {imu_data.accel_z / az if az != 0 else 0}")

            before = time.time()
        sim.get_world().tick()        

def main():
    # Initialize the Carla simulation
    sim = CarlaSimulation(town_name='Town07')
    
    sim.reset()
 
    
    # Add an ego vehicle at a specific position and rotation
    ego_vehicle = sim.add_ego_vehicle(pos=(0, 0, 2), rotation=(0, 0, 0), vehicle_type='vehicle.tesla.model3')
    ego_vehicle.attach_gps_sensor(50)
    ego_vehicle.attach_imu_sensor(50)
    
    # Set the pose of the ego vehicle
    #ego_vehicle.set_pose(MapPose(0, 0, 2))
    
    # Enable autopilot for the ego vehicle
    ego_vehicle.set_carla_autopilot(True)
    sim.move_spectator((0, 0, 20), (0, -10, 0))
    
    
    settings = sim.get_world().get_settings()
    settings.synchronous_mode = True
    sim.get_world().apply_settings(settings)
    
    
    read_sensor_thr = threading.Thread(target=sensor_show, daemon=True, kwargs={"sim": sim, "ego_vehicle": ego_vehicle})
    read_sensor_thr.start()
    
    print("Ego vehicle added and autopilot enabled.")
    input("Press Enter to exit...")
    
    settings = sim.get_world().get_settings()
    settings.synchronous_mode = False
    sim.get_world().apply_settings(settings)


if __name__ == "__main__":
    main()