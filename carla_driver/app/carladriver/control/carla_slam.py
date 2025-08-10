from pydriveless import MapPose, SLAM, angle

class CarlaSLAM (SLAM):
    __carla_ego_obj: any

    def __init__(self, carla_ego_obj: any):
        super().__init__()
        self.__carla_ego_obj = carla_ego_obj

    def calibrate(self) -> None:
        pass

    def estimate_ego_pose (self) -> MapPose:
        location = self.__carla_ego_obj.get_location()
        t = self.__carla_ego_obj.get_transform()
        heading = t.rotation.yaw
        pose = MapPose(x=location.x, y=location.y, z=location.z, heading=angle.new_deg(heading))
        return pose
    
