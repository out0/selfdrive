import sys
sys.path.append("../../../")
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
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

class FindGoalPoingDemo(QWidget):
    
    def __init__(self, file: str):
        super().__init__()
        self.setWindowTitle("Image Centered with Click Coordinates")
        self.resize(1024, 1024)

        bev_orig = np.array(cv2.imread(file))
        og = OccupancyGrid(
                frame=bev_orig,
                minimal_distance_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
                minimal_distance_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
    
        bev = og.get_color_frame()

        # Convert NumPy array to QImage
        height, width, channel = bev.shape
        bytes_per_line = 3 * width
        q_image = QImage(bev.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        self.image = QPixmap.fromImage(q_image)
        if self.image.isNull():
            print("Failed to create image from NumPy array.")
            sys.exit(1)

        # Image position
        self.image_center = QPoint(512, 512)
        self.image_top_left = QPoint(self.image_center.x() - bev.shape[0] // 2,
                                    self.image_center.y() - bev.shape[1] // 2)

        # To store last clicked position
        self.last_click_pos = None

        # Label to show click position (optional, we draw text directly)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 400, 30)
        self.label.setStyleSheet("color: black; font-size: 16px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background

        # Draw the image centered at (512,512)
        painter.drawPixmap(self.image_top_left, self.image)

        # If clicked, show the click coordinates
        if self.last_click_pos:
            text = f"Clicked at: ({self.last_click_pos.x()}, {self.last_click_pos.y()})"
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(10, 40, text)

    def mousePressEvent(self, event):
        self.last_click_pos = event.pos()
        self.update()  # trigger repaint

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FindGoalPoingDemo("bev_1.png")
    window.show()
    sys.exit(app.exec_())