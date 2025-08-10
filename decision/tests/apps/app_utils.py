import cv2, os, sys

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