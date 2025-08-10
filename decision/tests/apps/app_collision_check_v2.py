import sys
sys.path.append("../../")
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2, math

from app_utils import fix_cv2_import
fix_cv2_import()

from model.physical_paramaters import PhysicalParameters
from pydriveless import SearchFrame
from pydriveless import Waypoint
from pydriveless import WorldPose
from pydriveless import MapPose
from pydriveless import CoordinateConverter
from pydriveless import angle

COORD_ORIGIN = WorldPose(lat=-4.303359446566901e-09, 
                      lon=-1.5848012769283334e-08,
                      alt=1.0149892568588257,
                      compass=angle.new_rad(0))

class CollisionCheckDemo(QWidget):
    og: SearchFrame
    min_dist: tuple[int, int]

    def __init__(self, file: str):
        super().__init__()
        self.setWindowTitle("Image Centered with Click Coordinates")
        self.resize(1024, 1024)

        bev_orig = np.array(cv2.imread(file))
        self.og = SearchFrame(
                width=PhysicalParameters.OG_WIDTH,
                height=PhysicalParameters.OG_HEIGHT,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
        
        self.og.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
        self.og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        self.og.set_frame_data(bev_orig)
        
        # Image position
        self.image_top_left = QPoint(384, 384)


        # Label to show click position (optional, we draw text directly)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 400, 30)
        self.label.setStyleSheet("color: black; font-size: 16px;")
        
        self.min_dist = (PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        self.og.set_goal( min_distance=self.min_dist, x=128, z=0, compute_vectorized=False)
        self.min_dist_dir = int(round(math.sqrt(self.min_dist[0]**2 + self.min_dist[1]**2) / 2))


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background

        bev2 = self.og.get_color_frame()
        for z in range(bev2.shape[0]):
            for x in range(bev2.shape[1]):
                if not self.og.is_traversable(x,z):
                    bev2[z, x] = [0, 0, 0]

            # Convert NumPy array to QImage
            height, width, channel = bev2.shape
            bytes_per_line = 3 * width
            q_image = QImage(bev2.data, width, height, bytes_per_line, QImage.Format_RGB888)

            # Convert QImage to QPixmap
            self.image = QPixmap.fromImage(q_image)
            if self.image.isNull():
                print("Failed to create image from NumPy array.")
                sys.exit(1)



        # Draw the image centered at (512,512)

        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 16))
        painter.drawText(30, 100, f"Collision check v2 does not take angles into account. The min distance is {self.min_dist_dir} px in direction")
        painter.drawText(30, 130, f"Minimum Distances: {self.min_dist[0]} x {self.min_dist[1]} px")

        painter.drawPixmap(self.image_top_left, self.image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    #window = FindGoalPointDemo("bev_1.png")
    window = CollisionCheckDemo("bev_1.png")
    window.show()
    sys.exit(app.exec_())