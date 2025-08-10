import sys, time
import unittest, math
import os
import numpy as np

TOLERANCE = 0.001


SEGMENTED_COLORS = np.array([
        [0,   0,   0],
        [128,  64, 128],
        [244,  35, 232],
        [70,  70,  70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170,  30],
        [220, 220,   0],
        [107, 142,  35],
        [152, 251, 152],
        [70, 130, 180],
        [220,  20,  60],
        [255,   0,   0],
        [0,   0, 142],
        [0,   0,  70],
        [0,  60, 100],
        [0,  80, 100],
        [0,   0, 230],
        [119,  11,  32],
        [110, 190, 160],
        [170, 120,  50],
        [55,  90,  80],     # other
        [45,  60, 150],
        [157, 234,  50],
        [81,   0,  81],
        [150, 100, 100],
        [230, 150, 140],
        [180, 165, 180]
    ])

SEGMENTATION_CLASS_COST = np.array([
        -1,
        0,
        -1,
        -1,
        -1,
        -1,
        0,
        0,   # LAMP? investigate...
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1, # car
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        0,
        -1,
        0,
        0,
        0,
        0,
        -1
    ])




class AssertTolerance:
    
    @classmethod
    def assertAlmostEqual(cls, test: unittest.TestCase, a, b):
        diff = abs(a - b)
        if diff > TOLERANCE:
            test.fail(f"values {a} and {b} are not equal with a diff of {diff} > tolerance of {TOLERANCE}")
            

def fix_cv2_import():
    try:
        from cv2.version import ci_build, headless
        ci_and_not_headless = ci_build and not headless
    except:
        pass
    if sys.platform.startswith("linux") and ci_and_not_headless:
        if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
            os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if sys.platform.startswith("linux") and ci_and_not_headless:
        if "QT_QPA_FONTDIR" in os.environ:
            os.environ.pop("QT_QPA_FONTDIR")