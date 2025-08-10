import sys
sys.path.append("../../../")
import math
import cv2
from ensemble.planner.interpolator import Interpolator, PlanningData, PlanningResult, PlannerResultType
from ensemble.planner.overtaker import Overtaker
from ensemble.planner.hybrid_a import HybridAStar
from ensemble.planner.bi_rrt import BiRRTStar
from PyQt5.QtWidgets import QLineEdit
from pydriveless import CoordinateConverter, angle
from pygpd import GoalPointDiscover
from pydriveless import MapPose
from pydriveless import WorldPose
from pydriveless import Waypoint
from ensemble.model.physical_paramaters import PhysicalParameters
from pydriveless import SearchFrame
from app_utils import fix_cv2_import
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QCheckBox


fix_cv2_import()


TIMEOUT_MS = 5000

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
    goal_invalid: bool
    manual_goal: bool
    coord_conv: CoordinateConverter
    show_min_dist: bool
    show_path: list[Waypoint]
    show_path_valid: bool
    _last_bev: np.ndarray

    def __init__(self, file: str):
        super().__init__()

        self._last_bev = None
        self.coord_conv = CoordinateConverter(origin=COORD_ORIGIN,
                                         width=PhysicalParameters.OG_WIDTH,
                                         height=PhysicalParameters.OG_HEIGHT,
                                         perceptionWidthSize_m=PhysicalParameters.OG_REAL_WIDTH,
                                         perceptionHeightSize_m=PhysicalParameters.OG_REAL_HEIGHT)
        self.local_goal_discover = GoalPointDiscover(self.coord_conv)
        self.ego_pose = MapPose(x=0, y=0, z=0, heading=0.0)

        self.og = SearchFrame(
            width=PhysicalParameters.OG_WIDTH,
            height=PhysicalParameters.OG_HEIGHT,
            lower_bound=PhysicalParameters.EGO_LOWER_BOUND,
            upper_bound=PhysicalParameters.EGO_UPPER_BOUND)

        self.og.set_class_colors(PhysicalParameters.SEGMENTED_COLORS)
        self.og.set_class_costs(PhysicalParameters.SEGMENTATION_CLASS_COST)

        self.__setup_first_initialize()
        self.load_image(file)


    def __setup_first_initialize(self):
        self.setWindowTitle("Image Centered with Click Coordinates")
        self.resize(1024, 1024)

        # Image position
        self.image_top_left = QPoint(384, 384)
        # Label to show click position (optional, we draw text directly)
        self.label = QLabel(self)
        self.label.setGeometry(10, 10, 400, 30)
        self.label.setStyleSheet("color: black; font-size: 16px;")

        self.btn_select_file = QPushButton("Select BEV", self)
        self.btn_select_file.setGeometry(80, 60, 140, 30)
        self.btn_select_file.clicked.connect(self.btn_select_file_bev)

        self.btn_select_file = QPushButton("Export BEV", self)
        self.btn_select_file.setGeometry(230, 60, 140, 30)
        self.btn_select_file.clicked.connect(self.btn_export_to_output)

        self.btn_set_l1 = QPushButton("Set L1", self)
        self.btn_set_l1.setGeometry(80, 100, 70, 30)
        self.btn_set_l1.clicked.connect(self.btn_btn_set_l1_handler)

        self.txt_l1_x_input = QLineEdit(self)
        self.txt_l1_x_input.setGeometry(170, 100, 35, 30)
        self.txt_l1_x_input.setPlaceholderText("X")
        self.txt_l1_x_input.editingFinished.connect(
            self.txt_l1_x_input_handler)

        self.txt_l1_z_input = QLineEdit(self)
        self.txt_l1_z_input.setGeometry(215, 100, 35, 30)
        self.txt_l1_z_input.setPlaceholderText("Z")
        self.txt_l1_z_input.editingFinished.connect(
            self.txt_l1_z_input_handler)

        self.txt_l1_heading_input = QLineEdit(self)
        self.txt_l1_heading_input.setGeometry(260, 100, 80, 30)
        self.txt_l1_heading_input.setPlaceholderText("L1 heading")
        self.txt_l1_heading_input.editingFinished.connect(
            self.txt_l1_heading_input_handler)

        self.btn_set_l2 = QPushButton("Set L2", self)
        self.btn_set_l2.setGeometry(80, 140, 70, 30)
        self.btn_set_l2.clicked.connect(self.btn_btn_set_l2_handler)

        self.txt_l2_x_input = QLineEdit(self)
        self.txt_l2_x_input.setGeometry(170, 140, 35, 30)
        self.txt_l2_x_input.setPlaceholderText("X")
        self.txt_l2_x_input.editingFinished.connect(
            self.txt_l1_x_input_handler)

        self.txt_l2_z_input = QLineEdit(self)
        self.txt_l2_z_input.setGeometry(215, 140, 35, 30)
        self.txt_l2_z_input.setPlaceholderText("Z")
        self.txt_l2_z_input.editingFinished.connect(
            self.txt_l1_z_input_handler)

        self.txt_l2_heading_input = QLineEdit(self)
        self.txt_l2_heading_input.setGeometry(260, 140, 80, 30)
        self.txt_l2_heading_input.setPlaceholderText("L2 heading")
        self.txt_l2_heading_input.editingFinished.connect(
            self.txt_l2_heading_input_handler)

        self.txt_goal_x_input = QLineEdit(self)
        self.txt_goal_x_input.setGeometry(170, 180, 35, 30)
        self.txt_goal_x_input.setPlaceholderText("X")
        self.txt_goal_x_input.editingFinished.connect(
            self.txt_goal_x_input_handler)

        self.txt_goal_z_input = QLineEdit(self)
        self.txt_goal_z_input.setGeometry(215, 180, 35, 30)
        self.txt_goal_z_input.setPlaceholderText("Z")
        self.txt_goal_z_input.editingFinished.connect(
            self.txt_goal_z_input_handler)

        self.txt_goal_heading_input = QLineEdit(self)
        self.txt_goal_heading_input.setGeometry(260, 180, 80, 30)
        self.txt_goal_heading_input.setPlaceholderText("L2 heading")
        self.txt_goal_heading_input.editingFinished.connect(
            self.txt_goal_heading_input_handler)
        
        self.chk_show_min_distances = QCheckBox("Show minimal distances", self)
        self.chk_show_min_distances.setGeometry(80, 220, 200, 30)
        self.chk_show_min_distances.setChecked(False)
        self.chk_show_min_distances.stateChanged.connect(self.chk_show_min_distances_handler)
        self.__set_planning_buttons()


    def load_image(self, file: str) -> None:
        bev_orig = np.array(cv2.imread(file))
        self.og.set_frame_data(bev_orig)
        self.og.process_safe_distance_zone(
            (PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX), False)
        
        bev = self.og.get_color_frame()
        self.set_image(bev)

        self.L1 = None
        self.L2 = None
        self.set_l1 = False
        self.set_l2 = False
        self.goal = None
        self.goal_invalid = True
        self.manual_goal = False
        self.show_min_dist = False

        self.chk_show_min_distances.setChecked(False)
        self.show_path = None
        self.show_path_valid = False

    def chk_show_min_distances_handler(self) -> None:
        self.show_min_dist = self.chk_show_min_distances.isChecked()
        self.update()
    
    def btn_export_to_output(self):
        if self._last_bev is None: return
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Select BEV PNG File", "output.png", "PNG Files (*.png);;All Files (*)", options=options)
        if file_name:
            cv2.imwrite(file_name, self._last_bev)


    def btn_select_file_bev(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "PNG File", "", "PNG Files (*.png);;All Files (*)", options=options)
        if file_name:
            self.load_image(file_name)
            self.update()
            

    def set_image(self, bev: np.array) -> None:
        self._last_bev = bev
         # Convert NumPy array to QImage
        height, width, channel = bev.shape
        bytes_per_line = 3 * width
        q_image = QImage(bev.data, width, height,
                         bytes_per_line, QImage.Format_RGB888)

        # Convert QImage to QPixmap
        self.image = QPixmap.fromImage(q_image)
        if self.image.isNull():
            print("Failed to create image from NumPy array.")
            sys.exit(1)

    def __set_planning_buttons(self) -> None:
        x_start = 140
        w_size = 130
        spacing = 10
        self.btn_planner_ensemble = QPushButton("Ensemble", self)
        self.btn_planner_ensemble.setGeometry(x_start, 800, w_size, 30)
        self.btn_planner_ensemble.clicked.connect(
            self.btn_planner_ensemble_handler)

        x_start += w_size + spacing
        self.btn_planner_interpolator = QPushButton("Interpolator", self)
        self.btn_planner_interpolator.setGeometry(x_start, 800, w_size, 30)
        self.btn_planner_interpolator.clicked.connect(
            self.btn_planner_interpolator_handler)

        x_start += w_size + spacing
        self.btn_planner_interpolator = QPushButton("Overtaker", self)
        self.btn_planner_interpolator.setGeometry(x_start, 800, w_size, 30)
        self.btn_planner_interpolator.clicked.connect(
            self.btn_planner_overtaker_handler)

        x_start += w_size + spacing
        self.btn_planner_interpolator = QPushButton("Hybrid A*", self)
        self.btn_planner_interpolator.setGeometry(x_start, 800, w_size, 30)
        self.btn_planner_interpolator.clicked.connect(
            self.btn_planner_hybrid_handler)

        x_start += w_size + spacing
        self.btn_planner_interpolator = QPushButton("Bi-RRT*", self)
        self.btn_planner_interpolator.setGeometry(x_start, 800, w_size, 30)
        self.btn_planner_interpolator.clicked.connect(
            self.btn_planner_birrt_handler)

    def btn_planner_interpolator_handler(self) -> None:
        planner = Interpolator(self.coord_conv, max_exec_time_ms=TIMEOUT_MS)
        
        g1 = self.coord_conv.convert(self.ego_pose, self.goal)
        g2 = None
        if self.L2 is not None:
            g2 = self.coord_conv.convert(self.ego_pose, self.L2)
        
        planning_data = PlanningData(
            og=self.og,
            ego_location=self.ego_pose,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        )
        
        planning_data.set_local_goal(self.goal)
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        result = planner.get_result()
        
        self.show_path = result.path
        self.show_path_valid = result.result_type == PlannerResultType.VALID
        self.update()
    
    def btn_planner_overtaker_handler(self) -> None:
        planner = Overtaker(max_exec_time_ms=TIMEOUT_MS)
        
        g1 = self.coord_conv.convert(self.ego_pose, self.goal)
        g2 = None
        if self.L2 is not None:
            g2 = self.coord_conv.convert(self.ego_pose, self.L2)
        
        planning_data = PlanningData(
            og=self.og,
            ego_location=self.ego_pose,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        )
        
        planning_data.set_local_goal(self.goal)
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        result = planner.get_result()
        
        self.show_path = result.path
        self.show_path_valid = result.result_type == PlannerResultType.VALID
        self.update()

    def btn_planner_ensemble_handler(self) -> None:
        pass

    def btn_planner_birrt_handler(self) -> None:
        planner = BiRRTStar(map_coordinate_converter=self.coord_conv, 
                            max_exec_time_ms=TIMEOUT_MS,
                            max_path_size_px=30,
                            dist_to_goal_tolerance_px=5,
                            class_cost=PhysicalParameters.SEGMENTATION_CLASS_COST)
        
        g1 = self.coord_conv.convert(self.ego_pose, self.goal)
        g2 = None
        if self.L2 is not None:
            g2 = self.coord_conv.convert(self.ego_pose, self.L2)
        
        planning_data = PlanningData(
            og=self.og,
            ego_location=self.ego_pose,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        )
        
        planning_data.set_local_goal(self.goal)
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        result = planner.get_result()
        
        self.show_path = result.path
        self.show_path_valid = result.result_type == PlannerResultType.VALID
        self.update()

    def btn_planner_hybrid_handler(self) -> None:
        planner = HybridAStar(self.coord_conv, max_exec_time_ms=TIMEOUT_MS)
        
        g1 = self.coord_conv.convert(self.ego_pose, self.goal)
        g2 = None
        if self.L2 is not None:
            g2 = self.coord_conv.convert(self.ego_pose, self.L2)
        
        planning_data = PlanningData(
            og=self.og,
            ego_location=self.ego_pose,
            g1=g1,
            g2=g2,
            velocity=1.0,
            min_distance=(PhysicalParameters.MIN_DISTANCE_WIDTH_PX, PhysicalParameters.MIN_DISTANCE_HEIGHT_PX)
        )
        
        planning_data.set_local_goal(self.goal)
        self.og.process_distance_to_goal(self.goal.x, self.goal.z)
        planner.plan(planning_data)
        while planner.is_planning():
            pass
        result = planner.get_result()
        
        self.show_path = result.path
        self.show_path_valid = result.result_type == PlannerResultType.VALID
        self.update()



    def btn_btn_set_l1_handler(self):
        self.set_l1 = True

    def btn_btn_set_l2_handler(self):
        self.set_l2 = True

    def get_input_int(self, field: QLineEdit) -> int:
        try:
            return int(field.text())
        except ValueError:
            return 0.0

    def get_input_float(self, field: QLineEdit) -> float:
        try:
            return float(field.text())
        except ValueError:
            return 0.0

    def txt_l1_x_input_handler(self):
        x = self.get_input_int(self.txt_l1_x_input)
        if self.L1 is None:
            return
        new_l1 = Waypoint(x, self.L1.z, self.L1.heading)
        if new_l1 != self.L1:
            self.L1 = new_l1
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_l1_z_input_handler(self):
        z = self.get_input_int(self.txt_l1_z_input)
        if self.L1 is None:
            return
        new_l1 = Waypoint(self.L1.x, z, self.L1.heading)
        if new_l1 != self.L1:
            self.L1 = new_l1
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_l1_heading_input_handler(self):
        heading_deg = self.get_input_float(self.txt_l1_heading_input)
        if self.L1 is None:
            return
        new_l1 = Waypoint(self.L1.x, self.L1.z, angle.new_deg(heading_deg))
        if new_l1 != self.L1:
            self.L1 = new_l1
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_l2_x_input_handler(self):
        x = self.get_input_int(self.txt_l2_x_input)
        if self.L2 is None:
            return
        new_l2 = Waypoint(x, self.L2.z, self.L2.heading)
        if new_l2 != self.L2:
            self.L2 = new_l2
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_l2_z_input_handler(self):
        z = self.get_input_int(self.txt_l2_z_input)
        if self.L2 is None:
            return
        new_l2 = Waypoint(self.L2.x, z, self.L2.heading)
        if new_l2 != self.L2:
            self.L2 = new_l2
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_l2_heading_input_handler(self):
        heading_deg = self.get_input_float(self.txt_l2_heading_input)
        if self.L2 is None:
            return
        new_l2 = Waypoint(self.L2.x, self.L2.z, angle.new_deg(heading_deg))
        if new_l2 != self.L2:
            self.L2 = new_l2
            self.goal_invalid = True
            self.manual_goal = False
        self.update()

    def txt_goal_x_input_handler(self):
        x = self.get_input_int(self.txt_goal_x_input)
        if self.goal is None:
            z = 0
            h = angle.new_rad(0.0)
        else:
            z = self.goal.z
            h = self.goal.heading
        new_goal = Waypoint(x, z, h)
        if new_goal != self.goal:
            self.goal = new_goal
            self.goal_invalid = True
            self.manual_goal = True
        self.update()

    def txt_goal_z_input_handler(self):
        z = self.get_input_int(self.txt_goal_z_input)
        if self.goal is None:
            x = 0
            h = angle.new_rad(0.0)
        else:
            x = self.goal.x
            h = self.goal.heading

        new_goal = Waypoint(x, z, h)
        if new_goal != self.goal:
            self.goal = new_goal
            self.goal_invalid = True
            self.manual_goal = True
        self.update()

    def txt_goal_heading_input_handler(self):
        heading_deg = self.get_input_float(self.txt_goal_heading_input)
        h = angle.new_deg(heading_deg)
        if self.goal is None:
            x = 0
            z = 0
        else:
            x = self.goal.x
            z = self.goal.z
        new_goal = Waypoint(x, z, h)
        if new_goal != self.goal:
            self.goal = new_goal
            self.goal_invalid = True
            self.manual_goal = True
        self.update()

    def show_l1(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.L1 is not None:
            self.draw_arrow(painter, self.L1.x, self.L1.z,
                            self.L1.heading.rad())
            painter.setPen(QColor(80, 0, 80))
            painter.drawText(self.L1.x+378, self.L1.z+388, "X")

    def show_l2(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.L2 is not None:
            self.draw_arrow(painter, self.L2.x, self.L2.z,
                            self.L2.heading.rad())
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.L2.x+378, self.L2.z+388, "X")

    def show_goal(self, painter: QPainter) -> None:
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 14))
        if self.goal is None:
            painter.drawText(80, 200, f"[NO]Goal")
        else:
            painter.drawText(80, 200, f"Goal")
            # painter.drawText(80, 200, f"Goal: ({self.goal.x}, {self.goal.z}) heading: {self.goal.heading.deg():.2f}")
            self.draw_arrow(painter, self.goal.x,
                            self.goal.z, self.goal.heading.rad())
            painter.drawText(self.goal.x+378, self.goal.z+388, "X")

    def update_goal(self) -> None:
        self.txt_goal_x_input.setText(str(self.goal.x))
        self.txt_goal_z_input.setText(str(self.goal.z))
        self.txt_goal_heading_input.setText(f"{self.goal.heading.deg():.2f}")

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
        end_x = int(start_x + ARROW_LENGTH *
                    math.cos(heading_rad - (math.pi/2)))
        end_y = int(start_y + ARROW_LENGTH *
                    math.sin(heading_rad - (math.pi/2)))
        painter.drawLine(start_x, start_y, end_x, end_y)
        # Draw arrow head (simple triangle)
        angle = math.atan2(end_y - start_y, end_x - start_x)
        arrow_head_size = 8
        for side in [-1, 1]:
            side_angle = angle + side * math.radians(25)
            hx = int(end_x - arrow_head_size * math.cos(side_angle))
            hy = int(end_y - arrow_head_size * math.sin(side_angle))
            painter.drawLine(end_x, end_y, hx, hy)

    def __draw_min_dist(self, og: SearchFrame, bev: np.ndarray, x: int, z: int, heading_rad: float):
        c = math.cos(heading_rad)
        s = math.sin(heading_rad)

        min_x, min_z = self.og._last_min_dist[0]//2,self.og._last_min_dist[1]//2
        for i in range(-min_z, min_z + 1):
            for j in range(-min_x, min_x + 1):
                xl = int(j * c - i * s + x)
                zl = int(j * s + i * c + z)

                if xl < 0 or xl >= bev.shape[1]:
                    continue
                if zl < 0 or zl >= bev.shape[0]:
                    continue

                if og.is_obstacle(xl, zl):
                    bev[zl, xl] = [180, 28, 28]
                else:
                    bev[zl, xl] = [255, 254, 233]


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(
            255, 255, 255))  # White background

        if self.show_path is not None:
            bev = self.og.get_color_frame()  
            for p in self.show_path:
                if self.show_min_dist:
                    self.__draw_min_dist(self.og, bev, p.x, p.z, p.heading.rad())
            for p in self.show_path:
                if self.show_min_dist:
                    if p.x - 1 > 0: bev[p.z, p.x - 1] = [0, 0, 0]
                    if p.x + 1 < bev.shape[1]: bev[p.z, p.x + 1] = [0, 0, 0]
                    bev[p.z, p.x] = [0, 0, 0]
                else:
                    bev[p.z, p.x] = [255, 255, 255]            

            self.set_image(bev)

        # Draw the image centered at (512,512)
        painter.drawPixmap(self.image_top_left, self.image)

        if self.show_path is not None and not self.show_path_valid:
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 14))

            first_pos = 0
            for p in self.show_path:
                if not self.og.is_traversable(p.x, p.z, p.heading, precision_check=True):
                    break
                first_pos += 1
            painter.drawText(210, 700, f"Planner path is invalid - first collision in {self.show_path[first_pos]}")
            p = self.show_path[first_pos]
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(p.x+378, p.z+388, "X")


        self.show_l1(painter)
        self.show_l2(painter)


        if self.L1 and self.L2 and self.goal_invalid:
            g1 = self.coord_conv.convert(self.ego_pose, self.L1)
            g2 = self.coord_conv.convert(self.ego_pose, self.L2)

            if not self.manual_goal:
                #print(f"goal discovery on {self.L1}, {self.L2}")
                self.goal = self.local_goal_discover.find(
                    frame=self.og,
                    ego_pose=self.ego_pose,
                    g1=g1,
                    g2=g2
                )
            self.goal_invalid = False
            self.update_goal()

        self.show_goal(painter)

    def mousePressEvent(self, event):
        p = event.pos()
        if self.set_l1:
            heading = self.get_input_float(self.txt_l1_heading_input)
            self.L1 = Waypoint(p.x - 384, p.y - 384,
                               angle.new_deg(heading))
            self.set_l1 = False
            self.txt_l1_x_input.setText(str(p.x - 384))
            self.txt_l1_z_input.setText(str(p.y - 384))
        if self.set_l2:
            heading = self.get_input_float(self.txt_l2_heading_input)
            self.L2 = Waypoint(p.x - 384, p.y - 384,
                               angle.new_deg(heading))
            self.set_l2 = False
            self.txt_l2_x_input.setText(str(p.x - 384))
            self.txt_l2_z_input.setText(str(p.y - 384))

        self.goal_invalid = True
        self.update()  # trigger repaint


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # window = FindGoalPointDemo("bev_1.png")
    window = FindGoalPointDemo("bev_1.png")
    window.show()
    sys.exit(app.exec_())
