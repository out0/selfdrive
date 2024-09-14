# Author: Shing-Yan Loo (yan99033@gmail.com)
# Run extended Kalman filter to calculate the real-time vehicle location
# 
# Credit: State Estimation and Localization for Self-Driving Cars by Coursera
#   Please consider enrolling the course if you find this tutorial helpful and  
#   would like to learn more about Kalman filter and state estimation for 
#   self-driving cars in general.

import numpy as np
from slam.rotations import Quaternion, omega, skew_symmetric, angle_normalize
from model.sensor_data import GpsData
from model.sensor_data import IMUData
from model.map_pose import MapPose

class ExtendedKalmanFilter:
    def __init__(self):
        # State (position, velocity and orientation)
        self.p = np.zeros([3, 1])
        self.v = np.zeros([3, 1])
        self.q = np.array([1, 0, 0, 0]).reshape(4,1)  # quaternion

        # State covariance
        self.p_cov = np.zeros([9, 9])

        # Last updated timestamp (to compute the position
        # recovered by IMU velocity and acceleration, i.e.,
        # dead-reckoning)
        self.last_ts = 0

        # Gravity
        self.g = np.array([0, 0, -9.81]).reshape(3, 1)

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
        self.gnss_init_xyz = None
        self.initialized = False

    def is_initialized(self):
        return self.initialized
    
    def add_calibration_gnss_data(self, gnss: MapPose):
        if self.gnss_init_xyz is None:
            self.gnss_init_xyz = np.array([gnss.x, gnss.y, gnss.z])
            self.n_gnss_taken = 1
        else:
            self.gnss_init_xyz[0] += gnss.x
            self.gnss_init_xyz[1] += gnss.y
            self.gnss_init_xyz[2] += gnss.z
            self.n_gnss_taken += 1
    
    def calibrate(self):
        self.gnss_init_xyz /= self.n_gnss_taken
        # self.p[:, 0] = self.gnss_init_xyz
        # self.q[:, 0] = Quaternion().to_numpy()
        self.p = self.gnss_init_xyz

        # Low uncertainty in position estimation and high in orientation and 
        # velocity
        pos_var = 1
        orien_var = 1000
        vel_var = 1000
        self.p_cov[:3, :3] = np.eye(3) * pos_var
        self.p_cov[3:6, 3:6] = np.eye(3) * vel_var
        self.p_cov[6:, 6:] = np.eye(3) * orien_var
        self.initialized = True


    def get_location(self):
        """Return the estimated vehicle location

        :return: x, y, z position
        :rtype: list
        """
        return self.p.reshape(-1).tolist()

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
        imu_f = np.array([imu.accel_x, imu.accel_y, imu.accel_z]).reshape(3, 1)
        imu_w = np.array([imu.gyro_x, imu.gyro_y, imu.gyro_z]).reshape(3, 1)

        # IMU sampling time
        #delta_t = imu.timestamp - self.last_ts
        #self.last_ts = imu.timestamp

        # Update state with imu
        R = Quaternion(*self.q).to_mat()
        self.p = self.p + delta_t * self.v + 0.5 * delta_t * delta_t * (R @ imu_f + self.g)
        self.v = self.v + delta_t * (R @ imu_f + self.g)
        self.q = omega(imu_w, delta_t) @ self.q

        # Update covariance
        F = self._calculate_motion_model_jacobian(R, imu_f, delta_t)
        Q = self._calculate_imu_noise(delta_t)
        self.p_cov = F @ self.p_cov @ F.T + self.l_jac @ Q @ self.l_jac.T

    def _calculate_motion_model_jacobian(self, R, imu_f, delta_t):
        """derivative of the motion model function with respect to the state

        :param R: rotation matrix of the state orientation
        :type R: NumPy array
        :param imu_f: IMU xyz acceleration (force)
        :type imu_f: NumPy array
        """
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * delta_t
        F[3:6, 6:] = -skew_symmetric(R @ imu_f) * delta_t

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

    def correct_state_with_gnss(self, gnss: MapPose):
        """Given the estimated global location by gnss, correct
        the vehicle state

        :param gnss: global xyz position
        :type x: Gnss class (see car.py)
        """
        # Global position
        x = gnss.x
        y = gnss.y
        z = gnss.z

        # Kalman gain
        K = self.p_cov @ self.h_jac.T @ (np.linalg.inv(self.h_jac @ self.p_cov @ self.h_jac.T + self.var_gnss))

        # Compute the error state
        delta_x = K @ (np.array([x, y, z])[:, None] - self.p)

        # Correction
        self.p = self.p + delta_x[:3]
        self.v = self.v + delta_x[3:6]
        delta_q = Quaternion(axis_angle=angle_normalize(delta_x[6:]))
        self.q = delta_q.quat_mult_left(self.q)

        # Corrected covariance
        self.p_cov = (np.identity(9) - K @ self.h_jac) @ self.p_cov
