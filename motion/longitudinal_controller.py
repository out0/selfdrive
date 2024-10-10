
import numpy as np
import time
from utils.telemetry import Telemetry

class LongitudinalController:    
    __KP = 1.0
    __KI = 0.2
    __KD = 0.01

    _velocity_read: callable
    _power_actuator : callable
    _brake_actuator : callable
    _error_prev: float
    _error_I: float
    _error_D: float
    _desired_speed: float
    _prev_throttle: float
    _actuation_period_s: float
   
    def __init__(self,  actuation_period_ms: int, velocity_read: callable, power_actuator: callable, brake_actuator: callable) -> None:
        self._actuation_period_s = actuation_period_ms / 1000
        self._last_actuation_time = 0
        self._desired_speed = 0.0
        self._prev_throttle = 0.0
        self._velocity_read = velocity_read
        self._power_actuator = power_actuator
        self._brake_actuator = brake_actuator
        self._error_I = 0.0
        self._error_D = 0.0
        self._error_prev = 0.0

    def brake(self, brake_strenght: float) -> None:
        self._power_actuator(0)
        self._brake_actuator(brake_strenght)
        self._desired_speed = 0
        self._last_actuation_time = time.time()
    
    def __speed_set_loop(self, error: float) -> None:
        
        if error < -50:
            self._power_actuator(0.0)
            self._brake_actuator(1.0)
            return
        
        if error < -30:
            self._power_actuator(0.0)
            self._brake_actuator(0.5)
            return
        
        if error < -10:
            self._power_actuator(0.0)
            self._brake_actuator(0.3)
            return
        
        acc = LongitudinalController.__KP * error \
                + LongitudinalController.__KI * self._error_I\
                + LongitudinalController.__KD * self._error_D

        throttle = (np.tanh(acc) + 1)/2
        if throttle - self._prev_throttle > 0.1:
            throttle = self._prev_throttle + 0.1
        
        self._prev_throttle = throttle
        
        self._power_actuator(throttle)
    
    def loop(self, dt: float) -> None:
        current_speed = self._velocity_read()
        
        # Autobreak
        if current_speed == 0 and self._desired_speed == 0:            
            #self._brake_actuator(1.0)
            return
        
        error = self._desired_speed - current_speed

        self._error_I += error * dt
        self._error_D = (error - self._error_prev) / dt
        self._error_prev = error
        
        if time.time() - self._last_actuation_time >= self._actuation_period_s:
            self.__speed_set_loop(error)
            self._last_actuation_time = time.time()
            


    def set_speed(self, desired_speed: float) -> None:
        #print(f"[longitudial controller] set_speed({desired_speed})")
        self._desired_speed = desired_speed

    def destroy(self) -> None:
        if self._desired_speed != 0:
            self._desired_speed = 0.0

        self.brake(1.0)



