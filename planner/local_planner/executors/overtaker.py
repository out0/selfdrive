from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
from planner.local_planner.executors.waypoint_interpolator import WaypointInterpolator
from vision.occupancy_grid_cuda import GridDirection
import math, numpy as np

class OvertakerPlanner(LocalPathPlannerExecutor):

    INSIDE = 0  # 0000
    LEFT = 1    # 0001
    RIGHT = 2   # 0010
    BOTTOM = 4  # 0100
    TOP = 8     # 1000

    _plan_task: Thread
    _search: bool
    _planner_data: PlanningData
    _result: PlanningResult
    _step_size: int
    _mid_x: int
    _mid_z: int

    NAME = "overtaker"

    def __init__(self, 
                 max_exec_time_ms: int, 
                 step_size: int) -> None:
        
        super().__init__(max_exec_time_ms)
        self._step_size = step_size


    def plan(self, planner_data: PlanningData, partial_result: PlanningResult) -> None:
        self._mid_x = math.floor(planner_data.og.width() / 2)
        self._mid_z = math.floor(planner_data.og.height() / 2)
        self._planner_data = planner_data
        self._search = True
        self._result = partial_result
        self._plan_task = Thread(target=self.__perform_local_planning)
        self._plan_task.start()

    def cancel(self) -> None:
        self._search = False
        self._plan_task = None

    def is_planning(self) -> bool:
        return self._search

    def get_result(self) -> PlanningResult:
        return self._result

    def __compute_euclidian_distance(self, p1: Waypoint, p2: Waypoint) -> float:
        dz = p2.z - p1.z
        dx = p2.x - p1.x
        return math.sqrt(dz * dz + dx * dx)

    def __gen_intra_points(self, p1: Waypoint, p2: Waypoint) -> list[Waypoint]:
        lst: list[Waypoint] = []
        lst.append(p1)

        slope = math.atan2(p2.z - p1.z, p2.x - p1.x)
        size = self.__compute_euclidian_distance(p1, p2)
        path_step_size = size / self._step_size
        x_inc = path_step_size * math.cos(slope)
        z_inc = path_step_size * math.sin(slope)

        x = p1.x
        z = p1.z
        for _ in range(self._step_size - 1):
            x += x_inc
            z += z_inc

            lst.append(Waypoint(math.floor(x), math.floor(z)))

        lst.append(p2)
        return lst

    def __check_path_feasible(self, path: list[Waypoint]) -> bool:
        if path is None or len(path) <= 2:
            return False
        
        return self._planner_data.og.check_all_path_feasible(path)
    
    # def __dump_to_tmp(frame: np.ndarray, path: List[Waypoint], color = [255, 255, 255]):
    #     new_f = np.zeros(frame.shape)
    #     for i in range (frame.shape[0]):
    #         for j in range (frame.shape[1]):
    #             new_f[i, j, 0] = frame[i, j, 0]
    #             new_f[i, j, 1] = frame[i, j, 1]
    #             new_f[i, j, 2] = frame[i, j, 2]
                
    #     for p in path:
    #         frame[p.z, p.x, :] = color
    #     cv2.imwrite("overtake_tmp.png", frame)


    def __perform_local_planning(self) -> None:
        self.set_exec_started()
        self._rst_timeout()

        start = self._result.local_start
        goal = Waypoint(self._result.local_goal.x,
                        self._result.local_goal.z)

        
        path: list[Waypoint] = self.__gen_intra_points(start, goal)
        
        step = -1

        while self._search and not self._check_timeout():
            if self.__check_path_feasible(path):
                interpolated_path = WaypointInterpolator.interpolate_straight_line_path(
                    path[1], goal, self._og.height())

                if self.__check_path_feasible(interpolated_path):
                    self._result.result_type = PlannerResultType.VALID
                    self._result.path = interpolated_path
                    self._result.total_exec_time_ms = self.get_execution_time()
                    self._search = False
                    #OvertakerPlanner.__dump_to_tmp(self._og.get_color_frame(), path, color=[255, 0, 0])
                    return

            #OvertakerPlanner.__dump_to_tmp(self._og.get_color_frame(), path)

            valid, inc = self.__delocate_path(False,  step, path)

            if not valid:
                valid, inc = self.__delocate_path(True, step, path)
                if not valid:
                    break
            
            step += inc

        self._result.result_type = PlannerResultType.INVALID_PATH
        self._result.path = None
        self._result.timeout = self._search
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False

    def __delocate_path(self, reverse: bool, step: int, path: list[Waypoint]) -> tuple[bool, int]:
        n = len(path)
        w = self._og.width()
        
        if reverse:
            inc = 1
        else:
            inc = -1
        
        changes = np.zeros(n, dtype=np.int32)
        best_x = path[1].x
        for i in range (1, n):
            x = path[i].x
            z = path[i].z
            while (x >= 0 and x < w and not self._planner_data.og.check_direction_allowed(x, z, GridDirection.TOP)):
                x += inc
            
            if x < 0 or x >= w:
                if reverse:
                    return False, 0
                
                return self.__delocate_path(True, step, path)
        
            changes[i] = x
            
            if inc > 0:
                best_x = max(best_x, x)
            else:
                best_x = min(best_x, x)
            

        for i in range(1, n):
            path[i].x = best_x + step

        return True, inc

