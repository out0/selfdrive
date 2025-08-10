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