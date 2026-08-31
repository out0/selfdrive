"""
Qt6 app for generating and visualizing a Hermite-interpolated path
between two waypoints (p1, p2).

Requires: PyQt6, numpy, pywpmp, pydriveless
    pip install PyQt6 numpy

pywpmp / pydriveless are assumed to already be installed in your
environment (they are not public PyPI packages).
"""

import math
import sys

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pydriveless import Waypoint, angle
from pywpmp import HermiteInterpolator, KinematicInterpolator
from check_viable import check_not_reachable
from kinematic import kinematic_curve

CANVAS_W = 800
CANVAS_H = 800


class CurveApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hermite Curve Generator")
        self._last_image: QImage | None = None
        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ---- left: controls -------------------------------------------------
        controls = QVBoxLayout()

        p1_box = QGroupBox("P1")
        p1_form = QFormLayout(p1_box)
        self.p1_x = self._spin(0, CANVAS_W - 1, 400)
        self.p1_z = self._spin(0, CANVAS_H - 1, 750)
        self.p1_heading = self._spin(-360, 360, 0)
        p1_form.addRow("x", self.p1_x)
        p1_form.addRow("z", self.p1_z)
        p1_form.addRow("heading (deg)", self.p1_heading)
        controls.addWidget(p1_box)

        p2_box = QGroupBox("P2")
        p2_form = QFormLayout(p2_box)
        self.p2_x = self._spin(0, CANVAS_W - 1, 400)
        self.p2_z = self._spin(0, CANVAS_H - 1, 0)
        self.p2_heading = self._spin(-360, 360, 0)
        p2_form.addRow("x", self.p2_x)
        p2_form.addRow("z", self.p2_z)
        p2_form.addRow("heading (deg)", self.p2_heading)
        controls.addWidget(p2_box)

        params_box = QGroupBox("Curve parameters")
        params_form = QFormLayout(params_box)
        self.max_path_size_px = self._spin(0, 10000, 1000, step=10)
        self.turn_angle = self._spin(-360, 360, 40)
        self.wheelbase = self._spin(0.0, 20.0, 5.6, step=0.1)
        self.real_width = self._spin(1.0, CANVAS_W, 32, step=0.1)
        self.real_height = self._spin(1.0, CANVAS_H, 32, step=0.1)
        params_form.addRow("max path size (px)", self.max_path_size_px)
        params_form.addRow("steering (deg)", self.turn_angle)
        params_form.addRow("wheelbase (m)", self.wheelbase)
        params_form.addRow("real width (m)", self.real_width)
        params_form.addRow("real height (m)", self.real_height)
        controls.addWidget(params_box)

        btn_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Hermite")
        self.generate_btn.clicked.connect(self.generate_curve)
        self.save_btn = QPushButton("Save PNG…")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        btn_row.addWidget(self.generate_btn)
        btn_row.addWidget(self.save_btn)
        controls.addLayout(btn_row)

        btn_row2 = QHBoxLayout()
        self.generate2_btn = QPushButton("Generate Kinematic")
        self.generate2_btn.clicked.connect(self.generate_kinematic_curve)
        self.save2_btn = QPushButton("Save PNG…")
        self.save2_btn.clicked.connect(self.save_image)
        self.save2_btn.setEnabled(False)
        btn_row2.addWidget(self.generate2_btn)
        btn_row2.addWidget(self.save2_btn)
        controls.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self.generate3_btn = QPushButton("Generate P2 back curves")
        self.generate3_btn.clicked.connect(self.generate_kinematic_back_curves)
        btn_row3.addWidget(self.generate3_btn)
        controls.addLayout(btn_row3)

        btn_row4 = QHBoxLayout()
        self.generate4_btn = QPushButton("No reachable zone")
        self.generate4_btn.clicked.connect(self.show_no_reachable_zone)
        btn_row4.addWidget(self.generate4_btn)
        controls.addLayout(btn_row4)

        controls.addStretch(1)
        root.addLayout(controls, 0) 

        # ---- right: image preview -------------------------------------------
        self.image_label = QLabel("No curve generated yet")
        self.image_label.setFixedSize(CANVAS_W, CANVAS_H)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #111; color: #888;")
        root.addWidget(self.image_label, 1)

    @staticmethod
    def _spin(lo, hi, default, step=1):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(2)
        box.setSingleStep(step)
        box.setValue(default)
        return box

    def show_no_reachable_zone(self):
        try:
            goal = Waypoint(
                int(self.p2_x.value()),
                int(self.p2_z.value()),
                heading=angle.new_deg(self.p2_heading.value() + 180),
            )

            ratio_w = CANVAS_W / self.real_width.value()
            ratio_h = CANVAS_H / self.real_height.value()
            ratio_sq = math.sqrt(ratio_w * ratio_h)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            with open("outp1.dat", "w") as file:
                for z in range(CANVAS_H):
                    for x in range(CANVAS_W):
                        not_reachable = check_not_reachable(goal, angle.new_deg(40), int(self.wheelbase.value() * ratio_sq), x, z)
                        if not_reachable:
                            canvas[z, x, :] = [255, 0, 0]
                            file.write(f"({x}, {z})\n")

            image = QImage(
                canvas.data,
                CANVAS_W,
                CANVAS_H,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()  # copy() so the QImage owns its own buffer

            self._last_image = image
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.save_btn.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error generating curve", str(exc))        

    def accum(self, result: list, cx, cz, heading):
        result.append(Waypoint(cx, cz, angle.new_rad(heading)))
        return 1

    def generate_kinematic_back_curves(self):
        try:
            p1 = Waypoint(
                int(self.p2_x.value()),
                int(self.p2_z.value()),
                heading=angle.new_deg(self.p2_heading.value() + 180),
            )

            ratio_w = CANVAS_W / self.real_width.value()
            ratio_h = CANVAS_H / self.real_height.value()
            ratio_sq = math.sqrt(ratio_w * ratio_h)

            # res = []
            wheelbase_px = int(self.wheelbase.value() * ratio_sq)

            # interpolator = KinematicInterpolator()
            # curve1 = interpolator.kinematic_interpolation(CANVAS_W, 
            #                                      CANVAS_H, 
            #                                      p1, 
            #                                      angle.new_deg(self.turn_angle.value()),
            #                                      int(self.max_path_size_px.value()),
            #                                      wheelbase_px)
            # curve2 = interpolator.kinematic_interpolation(CANVAS_W, 
            #                                      CANVAS_H, 
            #                                      p1, 
            #                                      angle.new_deg(-self.turn_angle.value()),
            #                                      int(self.max_path_size_px.value()),
            #                                      wheelbase_px)
            curve1 = []
            kinematic_curve(
                (CANVAS_W, CANVAS_H),
                (p1.x, p1.z),
                p1.heading.rad(),
                math.radians(40),
                int(self.max_path_size_px.value()),
                wheelbase_px,
                self.accum,
                curve1)

            curve2 = []
            kinematic_curve(
                (CANVAS_W, CANVAS_H),
                (p1.x, p1.z),
                p1.heading.rad(),
                math.radians(-40),
                int(self.max_path_size_px.value()),
                wheelbase_px,
                self.accum,
                curve2)
            
            # RGB canvas, built directly (no cv2 / BGR conversion needed)
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            with open("outp2.dat", "w") as file:
                for points in [curve1, curve2]:
                    for p in points:
                        if 0 <= p.z < CANVAS_H and 0 <= p.x < CANVAS_W:
                            canvas[p.z, p.x, :] = [0, 255, 0]
                            file.write(f"({p.x}, {p.z})\n")   


            canvas[p1.z, p1.x, :] = [255, 255, 255]

            image = QImage(
                canvas.data,
                CANVAS_W,
                CANVAS_H,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()  # copy() so the QImage owns its own buffer

            self._last_image = image
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.save_btn.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error generating curve", str(exc))

    def generate_kinematic_curve(self):
        try:
            p1 = Waypoint(
                int(self.p1_x.value()),
                int(self.p1_z.value()),
                heading=angle.new_deg(self.p1_heading.value()),
            )

            ratio_w = CANVAS_W / self.real_width.value()
            ratio_h = CANVAS_H / self.real_height.value()
            ratio_sq = math.sqrt(ratio_w * ratio_h)

            res = []

            interpolator = KinematicInterpolator()
            res = interpolator.kinematic_interpolation(CANVAS_W, 
                                                 CANVAS_H, 
                                                 p1, 
                                                 angle.new_deg(self.turn_angle.value()),
                                                 int(self.max_path_size_px.value()),
                                                 int(self.wheelbase.value() * ratio_sq))

            # cost = kinematic_curve(
            #                 (CANVAS_W, CANVAS_H),
            #                 (p1.x, p1.z),
            #                 p1.heading.rad(),
            #                 math.radians(self.turn_angle.value()),
            #                 self.max_path_size_px.value(),
            #                 self.wheelbase.value() * ratio_sq,
            #                 self.callback_fn,
            #                 res
            #             )
            points = res

            # RGB canvas, built directly (no cv2 / BGR conversion needed)
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            for p in points:
                if 0 <= p.z < CANVAS_H and 0 <= p.x < CANVAS_W:
                    canvas[p.z, p.x, :] = [0, 255, 0]

            canvas[p1.z, p1.x, :] = [255, 255, 255]

            image = QImage(
                canvas.data,
                CANVAS_W,
                CANVAS_H,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()  # copy() so the QImage owns its own buffer

            self._last_image = image
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.save_btn.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error generating curve", str(exc))

    def generate_curve(self):
        try:
            p1 = Waypoint(int(self.p1_x.value()), int(self.p1_z.value()))
            p2 = Waypoint(
                int(self.p2_x.value()),
                int(self.p2_z.value()),
                heading=angle.new_deg(self.p2_heading.value()),
            )

            interpolator = HermiteInterpolator()
            points = interpolator.hermite_interpolation(
                CANVAS_W,
                CANVAS_H,
                p1,
                p2,
                1,
                math.radians(self.turn_angle.value()),
            )

            # RGB canvas, built directly (no cv2 / BGR conversion needed)
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

            for p in points:
                if 0 <= p.z < CANVAS_H and 0 <= p.x < CANVAS_W:
                    canvas[p.z, p.x, :] = [0, 255, 0]

            canvas[p1.z, p1.x, :] = [255, 255, 255]
            canvas[p2.z, p2.x, :] = [255, 255, 255]

            image = QImage(
                canvas.data,
                CANVAS_W,
                CANVAS_H,
                canvas.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()  # copy() so the QImage owns its own buffer

            self._last_image = image
            self.image_label.setPixmap(QPixmap.fromImage(image))
            self.save_btn.setEnabled(True)

        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Error generating curve", str(exc))

    def save_image(self):
        if self._last_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save curve image", "output.png", "PNG Image (*.png)"
        )
        if path:
            self._last_image.save(path, "PNG")


def main():
    app = QApplication(sys.argv)
    window = CurveApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()