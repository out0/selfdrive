#
# EKF - sensor fusion of IMU and GPS
#
# mapping over MapPose
#

import numpy as np
from utils.quaternion import Quaternion
from model.sensor_data import GpsData
from model.sensor_data import IMUData
from model.map_pose import MapPose
from model.world_pose import WorldPose
from data.coordinate_converter import CoordinateConverter
import math

class ExtendedKalmanFilter:
    
    _coord_converter: CoordinateConverter
    
    def __init__(self):
        # State (position, velocity and orientation)
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = Quaternion(1, 0, 0, 0)

        # State covariance
        self.p_cov = np.zeros([9, 9])

        # Last updated timestamp (to compute the position
        # recovered by IMU velocity and acceleration, i.e.,
        # dead-reckoning)
        self.last_ts = 0

        # Gravity
        self.g = np.array([0, 0, -9.81])

        # Sensor noise variances
        self.var_imu_acc = 0.01
        self.var_imu_gyro = 0.01

        # Motion model noise
        self.var_gnss = np.eye(3) * 100

        # Motion model noise Jacobian
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian

        # Measurement model Jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)

        # Initialized
        self.n_gnss_taken = 0
        self.gnss_init = None
        self.initialized = False
        self._coord_converter = None

    def is_initialized(self):
        return self.initialized
    
    def add_calibration_gnss_data(self, gnss: GpsData):
        
        # gnss_converted_to_map = self._coord_converter.convert_world_to_map_pose(
        #     WorldPose(
        #         lat=gnss.latitude,
        #         lon=gnss.longitude,
        #         alt=gnss.altitude,
        #         heading=0.0
        #     )
        # )
        
        if self.gnss_init is None:
            self.gnss_init = np.array([
                gnss.latitude,
                gnss.longitude,
                gnss.altitude
            ])
            self.n_gnss_taken = 1
        else:
            self.gnss_init[0] += gnss.latitude
            self.gnss_init[1] += gnss.longitude                
            self.gnss_init[2] += gnss.altitude
            self.n_gnss_taken += 1
    
    def calibrate(self, map_init_pos: MapPose = None):
        """_summary_

        Args:
            map_init_pos (MapPose, optional): sets an initial relative position.
        """
        self.gnss_init = self.gnss_init / self.n_gnss_taken
        
        world_origin = WorldPose(
                lat=self.gnss_init[0],
                lon=self.gnss_init[1],
                alt=self.gnss_init[2],
                heading=90.0
            )
        
        # first position computed from gnss mean
        self._coord_converter = CoordinateConverter(world_origin)
        if map_init_pos is None:
            map_init_pos = self._coord_converter.convert_world_to_map_pose(world_origin)
            
        self.p = np.array([
            map_init_pos.x,
            map_init_pos.y,
            0.0
        ])
        # self.p = np.array([
        #     map_init_pos.x,
        #     map_init_pos.y,
        #     map_init_pos.z
        # ])
        
        self.q = Quaternion.build_from_angles(np.array([0, 0, map_init_pos.heading]))       
        #self.q.rotate(0, 0, 1, math.radians(map_init_pos.heading))
        
        # Low uncertainty in position estimation and high in orientation and 
        # velocity
        pos_var = 1
        orien_var = 1000
        vel_var = 1000
        self.p_cov[:3, :3] = np.eye(3) * pos_var
        self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
        self.p_cov[6:, 6:] = np.eye(3) * orien_var
        self.initialized = True


    def get_location(self) -> MapPose:
        return MapPose(
            x=self.p[0],
            y=self.p[1],
            z=self.p[2],
            heading=self.q.get_yaw()
        )
        # """Return the estimated vehicle location

        # :return: x, y, z position
        # :rtype: list
        # """
        # return self.p.reshape(-1).tolist()

    def predict_state_with_imu(self, imu: IMUData, delta_t: float):
        """Use the IMU reading to update the car location (dead-reckoning)

        (This is equivalent to doing EKF prediction)

        Note that if the state is just initialized, there might be an error
        in the orientation that leads to incorrect state prediction. The error
        could be aggravated due to the fact that IMU is 'strapped down', and hence
        generating relative angular measurement (instead of absolute using IMU
        stabilized by a gimbal). Learn more in the Coursera course!

        The uncertainty (or state covariance) is going to grow larger and larger if there
        is no correction step. Therefore, the GNSS update would have a larger weight
        when performing the correction, and hopefully the state would converge toward
        the true state with more correction steps.

        :param imu: imu acceleration, velocity and timestamp
        :type imu: IMU blueprint instance (Carla)
        """
        # IMU acceleration and velocity
        imu_f = np.array([imu.accel_x, imu.accel_y, imu.accel_z, 1])
        #imu_w = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z, 1])
        
        imu_w = np.array([0.0, 0.0, imu.gyro_z, 1])

        # IMU sampling time
        #delta_t = imu.timestamp - self.last_ts
        #self.last_ts = imu.timestamp

        # Update state with imu
        R = Quaternion.build_mult_matrix(self.q)
        
        f = (imu_f @ R)[0:3]
        # f_no_gravity = f + self.g        
        # self.p = self.p + delta_t * self.v + 0.5 * delta_t * delta_t * f_no_gravity
        # self.v = self.v + delta_t * f_no_gravity

        f[2] = 0.0
        self.p = self.p + delta_t * self.v + 0.5 * delta_t * delta_t * f
        self.v = self.v + delta_t * f

        
        theta = imu_w * delta_t
        self.q = self.q * Quaternion.build_from_angles(theta)

        # Update covariance
        F = self._calculate_motion_model_jacobian(R, imu_f, delta_t)
        Q = self._calculate_imu_noise(delta_t)
        self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T


    def skew_symmetric(v):
        """Skew symmetric form of a 3x1 vector."""
        return np.array(
            [[0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]], dtype=np.float64)

    def _calculate_motion_model_jacobian(self, R, imu_f, delta_t):
        """derivative of the motion model function with respect to the state

        :param R: rotation matrix of the state orientation
        :type R: NumPy array
        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:] = -ExtendedKalmanFilter.skew_symmetric((R @ imu_f)[0:3] * delta_t)

        return F

    def _calculate_imu_noise(self, delta_t):
        """Calculate the IMU noise according to the pre-defined sensor noise profile

        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        :param imu_w: IMU xyz angular velocity
        :type imu_w: NumPy array
        """
        Q = np.eye(6)
        Q[:3, :3] *= delta_t * delta_t * self.var_imu_acc
        Q[3:, 3:] *= delta_t * delta_t * self.var_imu_gyro

        return Q

    def correct_state_with_gnss(self, gnss: GpsData, heading: float):
        """Given the estimated global location by gnss, correct
        the vehicle state

        :param gnss: global xyz position
        :type x: Gnss class (see car.py)
        """
        
        map_pose = self._coord_converter.convert_world_to_map_pose(WorldPose(
            lat=gnss.latitude,
            lon=gnss.longitude,
            alt=gnss.altitude,
            heading=heading
        ))
        
        # Global position
        x = map_pose.x
        y = map_pose.y
        z = map_pose.z

        # Kalman gain
        K = self.p_cov @ self.h_jac.T @ (np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + self.var_gnss))

        # Compute the error state
        delta_x = K @ (np.array([x, y, z]) - self.p).reshape((3,1))
        delta_x = delta_x.reshape(9)

        # Correction
        self.p = self.p + delta_x[:3]
        self.v = self.v + delta_x[3:6]
        delta_q = Quaternion.build_from_angles(delta_x[6:])
        self.q = delta_q * self.q

        # Corrected covariance
        self.p_cov = (np.identity(9) - K @ self.h_jac) @ self.p_cov

