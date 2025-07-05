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
        
        self.og = OccupancyGrid(
                frame=bev_orig,
                minimal_distance_x=14,
                minimal_distance_z=30,
                lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
                upper_bound=PhysicalParameters.EGO_UPPER_BOUND
            )
    
        bev = self.og.get_color_frame()

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
        self.image_top_left = QPoint(384, 384)


        # Label to show click position (optional, we draw text directly)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 400, 30)
        self.label.setStyleSheet("color: black; font-size: 16px;")
        
        self.L1 = None
        self.L2 = None
        self.click_first = True
        
        self.coord = CoordinateConverter(COORD_ORIGIN)
        self.local_goal_discover = GoalPointDiscover(self.coord)
        self.ego_pose = MapPose(x=0, y=0, z=0, heading=0.0)
        
  

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background

        # Draw the image centered at (512,512)
        painter.drawPixmap(self.image_top_left, self.image)

        # If clicked, show the click coordinates
        if self.L1:
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(10, 40, f"L1: ({self.L1.x}, {self.L1.z})")
            
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.L1.x+ 378, self.L1.z+ 388, "X")
            
        if self.L2:
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(10, 60, f"L2: ({self.L2.x}, {self.L2.z})")

            painter.setPen(QColor(0, 0, 255))
            painter.drawText(self.L2.x + 378, self.L2.z+ 388, "X")

        if self.L1 and self.L2:
            g1 = self.coord.convert_waypoint_to_map_pose(self.ego_pose, self.L1)
            g2 = self.coord.convert_waypoint_to_map_pose(self.ego_pose, self.L2)

            res = self.local_goal_discover.find_goal(
                og=self.og,
                current_pose=self.ego_pose,
                goal_pose=g1,
                next_goal_pose=g2
            )
            
            if res.goal is None:
                painter.setPen(QColor(0, 255, 0))
                painter.drawText(10, 80, "NO GOAL")
            else:
                painter.setPen(QColor(0, 255, 0))
                painter.drawText(10, 80, f"GOAL ({res.goal.x}, {res.goal.z}), heading: {res.goal.heading}")
            
                painter.setPen(QColor(0, 0, 0))
                painter.drawText(res.goal.x+378, res.goal.z+388, "X")
            


    def mousePressEvent(self, event):
        p = event.pos()
        if self.click_first:
            self.L1 = Waypoint(p.x() - 384, p.y() - 384)
            self.click_first = False
        else:
            self.L2 = Waypoint(p.x() - 384, p.y() - 384)
            self.click_first = True
            
        self.update()  # trigger repaint

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #window = FindGoalPointDemo("bev_1.png")
    window = FindGoalPointDemo("planning_data/bev_1.png")
    window.show()
    sys.exit(app.exec_())