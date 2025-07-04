import sys
sys.path.append("../../../")
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2, os
ci_build_and_not_headless = False
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    pass
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
if sys.platform.startswith("linux") and ci_and_not_headless:
    os.environ.pop("QT_QPA_FONTDIR")

from vision.occupancy_grid_cuda import OccupancyGrid
from model.physical_parameters import PhysicalParameters
from model.waypoint import Waypoint
from model.world_pose import WorldPose
from model.map_pose import MapPose
from planner.goal_point_discover import GoalPointDiscover
from data.coordinate_converter import CoordinateConverter

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      heading=0)

class FindGoalPointDemo(QWidget):
    
    L1: Waypoint
    L2: Waypoint
    click_first: bool
    og: OccupancyGrid
    angle: float
    old_angle: float
    
    
    def __init__(self, file: str):
        super().__init__()
        self.setWindowTitle("Image Centered with Click Coordinates")
        self.resize(1024, 1024)

        bev_orig = np.array(cv2.imread(file))
        self.og = OccupancyGrid(
                frame=bev_orig,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
        
        # Image position
        self.image_top_left = QPoint(384, 384)


        # Label to show click position (optional, we draw text directly)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 400, 30)
        self.label.setStyleSheet("color: black; font-size: 16px;")
        
        self.b_minus = QPushButton("(-)", self)
        self.b_minus.setGeometry(10, 50, 120, 30)
        self.b_minus.clicked.connect(self.btn_minus)

        self.b_plus = QPushButton("(+)", self)
        self.b_plus.setGeometry(140, 50, 120, 30)
        self.b_plus.clicked.connect(self.btn_plus)

        self.og.set_goal_vectorized(Waypoint(128, 0, 0))
        self.bev = self.og.get_cuda_frame().get_frame()

        self.angle = 0.0
        self.old_angle = None

    def btn_minus(self):
        if self.angle <= -67.5:
            return
        self.angle -= 22.5
        self.update()

    def btn_plus(self):
        if self.angle >= 90.0:
            return
        self.angle += 22.5
        self.update()


    def check_feasible(self, frame: np.ndarray, x: int, z: int, heading_deg: float):
        
        i = int(heading_deg/22.5) 
        a = 22.5 * i
        i += 3
        left = -1
        right = -1
            
        if heading_deg == a:
            left = i
        elif heading_deg > a:
            left = i
            right = i + 1
        else:
            left = i - 1
            right = i
            
        l = int(frame[z, x, 2])

        if left >= 0:
            if not (l & (1 << left)):
                return False
        if right >= 0:
            if not (l & (1 << right)):
                return False
            
        return True


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background

        if self.angle != self.old_angle:
            #bev2 = np.full(self.bev.shape, 0, dtype=np.uint8)
            bev2 = self.og.get_color_frame()
            for z in range(self.bev.shape[0]):
                for x in range(self.bev.shape[1]):
                    if not self.check_feasible(self.bev, x, z, self.angle):
                        bev2[z, x] = [0, 0, 0]
                    # if self.check_feasible(self.bev, x, z, self.angle):
                    #     bev2[z, x] = [255, 255, 255]

                    # if z >= PhysicalParameters.EGO_UPPER_BOUND.z and z <= PhysicalParameters.EGO_LOWER_BOUND.z:
                    #     if x >= PhysicalParameters.EGO_LOWER_BOUND.x and x <= PhysicalParameters.EGO_UPPER_BOUND.x:
                    #         bev2[z, x] = [255, 255, 255]


            # Convert NumPy array to QImage
            height, width, channel = self.bev.shape
            bytes_per_line = 3 * width
            q_image = QImage(bev2.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap
            self.image = QPixmap.fromImage(q_image)
            if self.image.isNull():
                print("Failed to create image from NumPy array.")
                sys.exit(1)

            self.old_angle = self.angle


        # Draw the image centered at (512,512)

        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 16))
        painter.drawText(30, 130, f"Current angle: {self.angle} deg")

        painter.drawPixmap(self.image_top_left, self.image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FindGoalPointDemo("bev_1.png")
    window.show()
    sys.exit(app.exec_())