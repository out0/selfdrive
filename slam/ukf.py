import numpy as np
import math
from model.sensors.imu import IMU, IMUData
from model.map_pose import MapPose
from utils.quaternion import Quaternion

class UKF:
    _gnss_init_data: np.ndarray
    _gnss_init_data_count: int
    _last_predicted_pos: MapPose
    _last_heading: Quaternion
    _state: np.ndarray

    def __init__(self) -> None:
        self._x = None
        self._gnss_init_data_count = 0
        self._last_predicted_pos = None
        self._last_heading = None
    
    def gnss_calibrate(self, gnss: MapPose):
        if self._gnss_init_data_count == 0:
            self._x = np.zeros(10)
    
        self._state[0] += gnss.x
        self._state[1] += gnss.y
        self._state[2] += gnss.z
        self._gnss_init_data_count += 1
    
    def calibrate(self):
        self._state /= self._gnss_init_data_count
        self._last_predicted_pos = MapPose(
            self._state[0],
            self._state[1],
            self._state[2],
            0
        )
        self._gnss_init_data_count = 0
        self._last_heading = Quaternion(1, 0, 0, 0)
    


    @classmethod
    def _motion_model(cls, dt: float, state: np.ndarray, imu_data: IMUData) -> np.ndarray:
        
        imu_f = np.array([imu_data.accel_x, imu_data.accel_y])
        imu_w = np.array([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])

        # pos += pos * last_V * dt
        state[0:2] += dt * state[2:4]  
        
        # v += 1/2 at²
        state[2:4] += 0.5 * dt * dt * imu_f

        theta = imu_w * dt

        diff_q = Quaternion.build_from_angles(theta)
        h = Quaternion(state[4], state[5], state[6], state[7])
        h = diff_q * h * diff_q.inv()

        state[4] = h.w
        state[5] = h.x
        state[6] = h.y
        state[7] = h.z

        return state

   
            
if __name__ == "__main__":
    x0 = np.array([0, 5])

    P0 = np.array([
    [0.00001, 0],
    [0, 1]
    ])
    Q = np.array([
            [0.01, 0],
            [0, 0.01]
        ])

    def motion_model(x: np.array) -> np.ndarray:
        A = np.array([
            [1, 0.5],
            [0, 1]
        ])
        return A @ x + -2 * np.array([0, 0.5])

    def meas_model(x) -> float:
        return math.atan(20 / (40 - x[0]))
        

    x1, P1 = UKF.predict(motion_model, x0, P0, Q)
                    

    X1c, P1c = UKF.correct(meas_model, x1, P1, math.pi/6, 0.01)
    
    print(f"estimate pos = {X1c[0]:.2f} m, velocity = {X1c[1]:.2f} m/s")
              
p = 1