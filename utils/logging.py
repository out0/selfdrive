
LOGGING_LEVEL = 3
LOGGING_FILE = "run.log"

from model.planning_data import PlanningData, PlanningResult, PlannerResultType
import cv2, os, numpy as np

PLANNING_DATA_PATH = "planning_data"

class Telemetry:
    
    @classmethod
    def log (cls, level: int, module: str, message: str) -> None:
        if level < LOGGING_LEVEL: return
        
        msg = f"[{module}] {message}"
        
        if LOGGING_FILE is None:
            print (msg)
        else:
            f = open(LOGGING_FILE, "a")
            f.write(f"{msg}\n")
            f.close()
    
    @classmethod
    def dump_planning_data(cls, level: int, seq: int, data: PlanningData, res: PlanningResult) -> None:
        if level > LOGGING_LEVEL: return
        
        if not os.path.exists (PLANNING_DATA_PATH):
            os.mkdir(PLANNING_DATA_PATH)
            
        with open(f"{PLANNING_DATA_PATH}/planning_result_{seq}.dat", "w") as f:
            f.write(str(res))

        cv2.imwrite(f"{PLANNING_DATA_PATH}/bev_{seq}.png", data.bev)   
        pass

    @classmethod
    def read_planning_result(cls, seq: int) -> PlanningResult:
        file = f"{PLANNING_DATA_PATH}/planning_result_{seq}.dat"

        if not os.path.exists (file):
            return None
            
        with open(file, "r") as f:
            return PlanningResult.from_str(f.read())

    @classmethod
    def read_planning_bev(cls, seq: int) -> PlanningData:
        file = f"{PLANNING_DATA_PATH}/bev_{seq}.png"

        if not os.path.exists (file):
            return None
            
        return np.array(cv2.imread(file))