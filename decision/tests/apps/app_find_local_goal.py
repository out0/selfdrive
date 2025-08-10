import sys
sys.path.append("../../../")
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2, math

from app_utils import fix_cv2_import
fix_cv2_import()

from pydriveless import SearchFrame
from ensemble.model.physical_paramaters import PhysicalParameters
from pydriveless import Waypoint
from pydriveless import WorldPose
from pydriveless import MapPose
from pygpd import GoalPointDiscover
from pydriveless import CoordinateConverter, angle
from PyQt5.QtWidgets import QLineEdit
from PyQt5.QtWidgets import QLineEdit



COORD_ORIGIN = WorldPose(lat=angle.new_deg(-4.303359446566901e-09), 
                      lon=angle.new_deg(-1.5848012769283334e-08),
                      alt=angle.new_deg(1.0149892568588257),
                      compass=angle.new_rad(0))

ARROW_LENGTH = 30

class FindGoalPointDemo(QWidget):
    
    L1: Waypoint
    L2: Waypoint
    goal: Waypoint
    set_l1: bool
    set_l2: bool
    og: SearchFrame
    
    def __load_format(self, bev_orig: np.array):
        self.setWindowTitle("Image Centered with Click Coordinates")
        self.resize(1024, 1024)
        
        self.og = SearchFrame(
            width=PhysicalParameters.OG_WIDTH,
            height=PhysicalParameters.OG_HEIGHT,
            lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
            upper_bound=PhysicalParameters.EGO_UPPER_BOUND)
        
        self.og.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
        self.og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)
        self.og.set_frame_data(bev_orig)
        self.og.process_safe_distance_zone((PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX), False)
        
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
        
        self.btn_set_l1 = QPushButton("Set L1", self)
        self.btn_set_l1.setGeometry(80, 100, 70, 30)
        self.btn_set_l1.clicked.connect(self.btn_btn_set_l1_handler)
        
        self.txt_l1_heading_input = QLineEdit(self)
        self.txt_l1_heading_input.setGeometry(170, 100, 80, 30)
        self.txt_l1_heading_input.setPlaceholderText("L1 heading")
        self.txt_l1_heading_input.editingFinished.connect(self.txt_l1_heading_input_handler)
        
        self.btn_set_l2 = QPushButton("Set L2", self)
        self.btn_set_l2.setGeometry(80, 140, 70, 30)
        self.btn_set_l2.clicked.connect(self.btn_btn_set_l2_handler)

        self.txt_l2_heading_input = QLineEdit(self)
        self.txt_l2_heading_input.setGeometry(170, 140, 80, 30)
        self.txt_l2_heading_input.setPlaceholderText("L2 heading")
        self.txt_l2_heading_input.editingFinished.connect(self.txt_l2_heading_input_handler)

        self.btn_log_coord = QPushButton("log coord.", self)
        self.btn_log_coord.setGeometry(80, 180, 120, 30)
        self.btn_log_coord.clicked.connect(self.btn_log_coord_handler)
        
        self.set_l1 = False
        self.set_l2 = False
        self.goal = None
        
    
    def __init__(self, file: str):
        super().__init__()

        bev_orig = np.array(cv2.imread(file))
        self.__load_format(bev_orig)

        self.L1 = None
        self.L2 = None
        self.click_first = True
        
        self.coord = CoordinateConverter(origin=COORD_ORIGIN, 
                                         width=PhysicalParameters.OG_WIDTH,
                                         height=PhysicalParameters.OG_HEIGHT,
                                         perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH,
                                         perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT)
        self.local_goal_discover = GoalPointDiscover(self.coord)
        self.ego_pose = MapPose(x=0, y=0, z=0, heading=0.0)
        
    def btn_btn_set_l1_handler(self):
        self.set_l1 = True
    
    def btn_btn_set_l2_handler(self):
        self.set_l2 = True
    
    def get_input_float(self, field: QLineEdit) -> float:
        try:
            return float(field.text())
        except ValueError:
            return 0.0
  
    def txt_l1_heading_input_handler(self):
        heading_deg = self.get_input_float(self.txt_l1_heading_input)
        if self.L1 is None: return
        self.L1 = Waypoint(self.L1.x, self.L1.z, angle.new_deg(heading_deg))
        self.update()
  
    def txt_l2_heading_input_handler(self):
        heading_deg = self.get_input_float(self.txt_l2_heading_input)
        if self.L2 is None: return
        self.L2 = Waypoint(self.L2.x, self.L2.z, angle.new_deg(heading_deg))
        self.update()
    
    def show_l1(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.L1 is None:
            painter.drawText(270, 120, f"L1: not set")
        else:
            painter.drawText(270, 120, f"L1: ({self.L1.x}, {self.L1.z}) heading: {self.L1.heading.deg():.2f}") 
            self.draw_arrow(painter, self.L1.x, self.L1.z, self.L1.heading.rad())
            painter.setPen(QColor(80, 0, 80))
            painter.drawText(self.L1.x+378, self.L1.z+388, "X")
            
    def show_l2(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.L2 is None:
            painter.drawText(270, 160, f"L2: not set")
        else:
            painter.drawText(270, 160, f"L2: ({self.L2.x}, {self.L2.z}) heading: {self.L2.heading.deg():.2f}")
            self.draw_arrow(painter, self.L2.x, self.L2.z, self.L2.heading.rad())
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.L2.x+378, self.L2.z+388, "X")
    
    def show_goal(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.goal is None:
            painter.drawText(270, 200, f"Goal: not found")
        else:
            painter.drawText(270, 200, f"Goal: ({self.goal.x}, {self.goal.z}) heading: {self.goal.heading.deg():.2f}")
            self.draw_arrow(painter, self.goal.x, self.goal.z, self.goal.heading.rad())
            painter.drawText(self.goal.x+378, self.goal.z+388, "X")
     
    def btn_log_coord_handler(self):
        if self.L1 and self.L2:
            with open("find_local_goal.log", "a") as f:
                f.write(f"{self.L1.x},{self.L1.z},{self.L1.heading.deg()},{self.L2.x},{self.L2.z},{self.L2.heading.deg()},{self.last_res.x},{self.last_res.z},{self.last_res.heading.deg()}\n")
    def draw_arrow(self, painter: QPainter, x: int, y: int, heading_rad: float):
        arrow_color = QColor(0, 0x72, 0)
        painter.setPen(arrow_color)
        painter.setBrush(arrow_color)
        
        # Arrow start point (centered on image)
        start_x = x + 384
        start_y = y + 384
        # Arrow end point
        end_x = int(start_x + ARROW_LENGTH * math.cos(heading_rad - (math.pi/2)))
        end_y = int(start_y + ARROW_LENGTH * math.sin(heading_rad - (math.pi/2)))
        painter.drawLine(start_x, start_y, end_x, end_y)
        # Draw arrow head (simple triangle)
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_head_size = 8
        for side in [-1, 1]:
            side_angle = angle + side * math.radians(25) 
            hx = int(end_x - arrow_head_size * math.cos(side_angle))
            hy = int(end_y - arrow_head_size * math.sin(side_angle))
            painter.drawLine(end_x, end_y, hx, hy)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255))  # White background

        # Draw the image centered at (512,512)
        painter.drawPixmap(self.image_top_left, self.image)

        self.show_l1(painter)
        self.show_l2(painter)

                
        if self.L1 and self.L2:
            g1 = self.coord.convert(self.ego_pose, self.L1)
            g2 = self.coord.convert(self.ego_pose, self.L2)

            self.goal = self.local_goal_discover.find(
                frame=self.og,
                ego_pose=self.ego_pose,
                g1=g1,
                g2=g2
            )
            
        self.show_goal(painter)            

    def mousePressEvent(self, event):
        p = event.pos()
        if self.set_l1:
            heading = self.get_input_float(self.txt_l1_heading_input)
            self.L1 = Waypoint(p.x - 384, p.y - 384, angle.new_deg(heading))
            self.set_l1 = False
        if self.set_l2:
            heading = self.get_input_float(self.txt_l2_heading_input)
            self.L2 = Waypoint(p.x - 384, p.y - 384, angle.new_deg(heading))
            self.set_l2 = False    
        self.update()  # trigger repaint

if __name__ == "__main__":
    app = QApplication(sys.argv)
    #window = FindGoalPointDemo("bev_1.png")
    window = FindGoalPointDemo("bev_1.png")
    window.show()
    sys.exit(app.exec_())