import sys
sys.path.append("../../../")
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QCloseEvent, QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QLineEdit
import cv2, math
from pydriveless import DiscreteComponent, angle
from carladriver import CarlaEgoVehicle, CarlaSimulation
from ensemble.motion.longitudinal_controller import LongitudinalController
import time
from app_utils import fix_cv2_import
fix_cv2_import()


class SimpleVelocityKeeper(DiscreteComponent):
    _controller: LongitudinalController
    _ego: CarlaEgoVehicle

    def __init__(self, ego: CarlaEgoVehicle, period_ms: int):
        super().__init__(period_ms)
        self._ego = ego

        self._longitudinal_controller = LongitudinalController(
            period_ms,
            brake_actuator=self.__set_brake,
            power_actuator=self.__set_power,
            velocity_read=self.__get_velocity
        )
    
    def __set_brake(self, level: float) -> None:
        self._ego.set_brake(level)

    def __set_power(self, level: float) -> None:
        self._ego.set_power(level)

    def __get_velocity(self) -> float:
        return self._ego.read_odometer()

    def _loop(self, dt: float) -> None: 
        self._longitudinal_controller.loop(dt)
    
    def set_velocity(self, val: float) -> None:
        self._longitudinal_controller.brake(0.0)
        self._longitudinal_controller.set_speed(val / 3.6)

    def get_velocity(self) -> float:
        return 3.6 * self._ego.read_odometer()

    def brake(self) -> None:
        self._longitudinal_controller.set_speed(0)
        self._longitudinal_controller.brake(1.0)


class CarController(QWidget):

    _velocity_keeper: SimpleVelocityKeeper
    _ego: CarlaEgoVehicle
    _current_velocity = 0.0
    _simulation: CarlaSimulation
    _steering_angle: int
    _power_factor: float

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Controller")
        #self.setGeometry(100, 100, 600, 200)
        self.resize(470, 200)
        self.center_on_screen()
        self.__speed_buttons()
        self.__direction_controls()
        self._simulation = CarlaSimulation(town_name="Town07")
        self._ego = self._simulation.add_ego_vehicle(pos=[0, 0, 2])
        #self._velocity_keeper = SimpleVelocityKeeper(self._ego, period_ms=10)
        self._steering_angle = 0
        self._power_factor = 0.0
        #self._velocity_keeper.start()


    def closeEvent(self, event: QCloseEvent):
        #self._velocity_keeper.brake()
        #self._velocity_keeper.destroy()
        self._ego.destroy()

    def center_on_screen(self):
        frame_geometry = self.frameGeometry()
        screen_center = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())        

    def __speed_buttons(self) -> None:
        self.btn_speed_up = QPushButton("V (+)", self)
        self.btn_speed_up.setGeometry(50, 50, 80, 30)
        self.btn_speed_up.clicked.connect(self.btn_speed_up_handler)

        self.btn_speed_down = QPushButton("V (-)", self)
        self.btn_speed_down.setGeometry(50, 90, 80, 30)
        self.btn_speed_down.clicked.connect(self.btn_speed_down_handler)

    def __direction_controls(self) -> None:
        x_init = 170
        self.btn_dir_left = QPushButton("<--", self)
        self.btn_dir_left.setGeometry(x_init, 90, 80, 30)
        self.btn_dir_left.clicked.connect(self.btn_left_handler)

        self.label_d = QLabel(self)
        self.label_d.setGeometry(x_init + 90, 50, 70, 30)
        self.label_d.setStyleSheet("color: black; font-size: 16px;")
        self.label_d.setText("12 deg")
        self.label_d.setAlignment(Qt.AlignCenter)

        self.label_v = QLabel(self)
        self.label_v.setGeometry(x_init + 90, 90, 70, 30)
        self.label_v.setStyleSheet("color: black; font-size: 16px;")
        self.label_v.setText("0 km/h")  
        self.label_v.setAlignment(Qt.AlignCenter)

        self.btn_dir_right = QPushButton("-->", self)
        self.btn_dir_right.setGeometry(x_init + 90 + 80, 90, 80, 30)
        self.btn_dir_right.clicked.connect(self.btn_right_handler)

        self.btn_brake = QPushButton(" Brake ", self)
        self.btn_brake.setGeometry(50, 140, 372, 30)
        self.btn_brake.clicked.connect(self.btn_brake_handler)


    def btn_speed_up_handler(self):
        self._power_factor += 0.1
        if self._power_factor > 1.0:
            self._power_factor = 1.0
        #self._current_velocity += 2.0
        #self._velocity_keeper.set_velocity(self._current_velocity)
        self._ego.set_brake(0.0)
        self._ego.set_power(self._power_factor)
        self.update()

    def btn_speed_down_handler(self):
        self._power_factor -= 0.1
        if self._power_factor < -1.0:
            self._power_factor = -1.0
        self._ego.set_brake(0.0)
        self._ego.set_power(self._power_factor)
        # self._current_velocity -= 2.0
        # self._velocity_keeper.set_velocity(self._current_velocity)
        self.update()

    def btn_left_handler(self):
        self._steering_angle -= 1
        if self._steering_angle < -40:
            self._steering_angle = -40
        self._ego.set_steering(self._steering_angle)
        self.update()

    def btn_right_handler(self):
        self._steering_angle += 1
        if self._steering_angle > 40:
            self._steering_angle = 40
        self._ego.set_steering(self._steering_angle)
        self.update()

    def btn_brake_handler(self):
        # self._current_velocity = 0
        # self._velocity_keeper.brake()
        self._power_factor = 0
        self._ego.set_power(0.0)
        self._ego.set_brake(1.0)
        self.update()

    def btn_rst_steering(self):
        self._steering_angle = 0
        self._ego.set_steering(self._steering_angle)
        self.update()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_W:
            self.btn_speed_up.click()
        elif event.key() == Qt.Key_S:
            self.btn_speed_down.click()
        elif event.key() == Qt.Key_A:
            self.btn_dir_left.click()
        elif event.key() == Qt.Key_D:
            self.btn_dir_right.click()
        elif event.key() == Qt.Key_E:
            self.btn_brake.click()
        elif event.key() == Qt.Key_Q:
            self.btn_rst_steering()

    def paintEvent(self, event):
        painter = QPainter(self)
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #window = FindGoalPointDemo("bev_1.png")
    window = CarController()
    window.show()
    sys.exit(app.exec_())