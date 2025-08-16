import sys, time
import unittest, math
import os

TOLERANCE = 0.001

class AssertTolerance:
    
    @classmethod
    def assertAlmostEqual(cls, test: unittest.TestCase, a, b):
        diff = abs(a - b)
        if diff > TOLERANCE:
            test.fail(f"values {a} and {b} are not equal with a diff of {diff} > tolerance of {TOLERANCE}")
            

def fix_cv2_import():
    import cv2
    # Safe check for OpenCV build flags
    ci_and_not_headless = getattr(cv2, "ci_build", False) and not getattr(cv2, "headless", False)

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

    if sys.platform.startswith("linux") and ci_and_not_headless:
        os.environ.pop("QT_QPA_FONTDIR", None)