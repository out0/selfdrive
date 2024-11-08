from threading import Thread, Lock
from queue import Queue
from model.planning_data import PlanningData, PlanningResult, PlannerResultType, CollisionReport
import cv2, os, numpy as np
import json
from model.map_pose import MapPose
PLANNING_DATA_PATH = "planning_data"

class TelemetrySession:
    __planning_data_path: str
    __exec_seq: int
    __mtx: Lock
    __dump_data: dict

    def __init__(self, planning_data_path: str) -> None:
        self.__planning_data_path = planning_data_path
        self.__exec_seq = 0
        self.__mtx = Lock()
        pass

    def log_pre_planning_data(self, data: PlanningData) -> None:
        self.__mtx.acquire(blocking=True)
        self.__exec_seq += 1
        self.__dump_data = {
            'pre_planning_result' : PlanningResult(
                planner_name="-",
                local_start = None,
                local_goal = None,
                goal = data.goal,
                next_goal = data.next_goal,
                direction = 0,
                ego_location = data.ego_location,
                timeout=False,
                path=None,
                result_type=PlannerResultType.NONE,
                total_exec_time_ms=0
            ),
            'pre_planning_bev' : data.bev
        }
        self.__mtx.release()

    def log_planning_data(self, data: PlanningData, res: PlanningResult) -> None:
        self.__mtx.acquire(blocking=True)
        self.__dump_data['planning_result'] = res
        self.__dump_data['planning_bev'] = data.bev
        self.__mtx.release()
        self.__terminate()
        
    def log_collision(self, report: CollisionReport,  collision_bev: np.ndarray) -> None:
        self.__mtx.acquire(blocking=True)
        self.__dump_data['collision_detected'] = report
        self.__dump_data['collision_bev'] = collision_bev
        self.__mtx.release()
        self.__terminate()
        
    def __terminate(self):
        self.__mtx.acquire(blocking=True)
        self.__save_to_disk()
        
    def __save_to_disk(self):
        thr = Thread(target=self.__save_to_disk_thr)
        thr.start()
        pass


    def __save_to_disk_thr(self):
        dump_data_save = self.__dump_data
        seq = self.__exec_seq
        self.__mtx.release()
        
        if not os.path.exists (self.__planning_data_path):
            os.mkdir(self.__planning_data_path)
            
        log_file = f"{self.__planning_data_path}/planning_result_{seq}.json"
           
        
        cv2.imwrite(f"{self.__planning_data_path}/pre_bev_{seq}.png", dump_data_save['pre_planning_bev'])
        cv2.imwrite(f"{self.__planning_data_path}/bev_{seq}.png", dump_data_save['planning_bev'])
        pre_result = str(dump_data_save['pre_planning_result'])
        result = str(dump_data_save['planning_result'])

        if 'collision_detected' in dump_data_save:
            collision = str(dump_data_save['collision_detected'])
        else:
            collision = ""
                   
        outp = {
            "pre_planning" : pre_result,
            "planning": result,
            "collision_report": collision
        }
        
        with open(log_file, "w") as f:
            f.write(f"{json.dumps(outp)}\n")
        
        if 'collision_bev' in dump_data_save:
            cv2.imwrite(f"{self.__planning_data_path}/collision_bev_{seq}.png", dump_data_save['collision_bev'])

class Telemetry:
    __telemetry_session: TelemetrySession

    @classmethod
    def initialize(cls) -> None:
        Telemetry.__telemetry_session = TelemetrySession(
            planning_data_path = PLANNING_DATA_PATH
        )   

    def log_pre_planning_data(data: PlanningData) -> None:
        Telemetry.__telemetry_session.log_pre_planning_data(data)

    def log_planning_data(data: PlanningData, res: PlanningResult) -> None:
        Telemetry.__telemetry_session.log_planning_data(data, res)
        
    def log_collision(report: CollisionReport, collision_bev: np.ndarray) -> None:
        Telemetry.__telemetry_session.log_collision(report, collision_bev)

    @classmethod
    def read_collision_report(cls, seq: int) -> CollisionReport:
        file = f"{PLANNING_DATA_PATH}/planning_result_{seq}.json"
        if not os.path.exists (file):
            return None
        
        with open(file, "r") as f:
            j = json.loads(f.read())
            return CollisionReport.from_str(j['collision_report'])


    @classmethod
    def read_pre_planning_result(cls, seq: int) -> PlanningResult:
        file = f"{PLANNING_DATA_PATH}/planning_result_{seq}.json"

        if not os.path.exists (file):
            return None
            
        with open(file, "r") as f:
            j = json.loads(f.read())
            return PlanningResult.from_str(j["pre_planning"])
    
    
    @classmethod
    def read_planning_result(cls, seq: int) -> PlanningResult:
        file = f"{PLANNING_DATA_PATH}/planning_result_{seq}.json"

        if not os.path.exists (file):
            return None
            
        with open(file, "r") as f:
            j = json.loads(f.read())
            return PlanningResult.from_str(j["planning"])
        
    @classmethod
    def read_planning_result_from_file(cls, file: str) -> PlanningResult:
        if not os.path.exists (file):
            return None
            
        with open(file, "r") as f:
            j = json.loads(f.read())
            return PlanningResult.from_str(j["planning"])

    @classmethod
    def __read_bev(cls, file: str) -> PlanningResult:
        if not os.path.exists (file):
            return None
            
        return np.array(cv2.imread(file))

    @classmethod
    def read_planning_bev(cls, seq: int) -> np.ndarray:
        file = f"{PLANNING_DATA_PATH}/bev_{seq}.png"
        return cls.__read_bev(file)
    
    @classmethod
    def read_pre_planning_bev(cls, seq: int) -> np.ndarray:
        file = f"{PLANNING_DATA_PATH}/pre_bev_{seq}.png"
        return cls.__read_bev(file)
        
    
    @classmethod
    def read_collision_bev(cls, seq: int) -> np.ndarray:
        file = f"{PLANNING_DATA_PATH}/collision_bev_{seq}.png"
        return cls.__read_bev(file)