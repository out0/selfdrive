from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
from model.physical_parameters import PhysicalParameters
#from planner.local_planner.executors.cpu_rrt import RRT
from planner.local_planner.executors.cpu_rrt_bidirectional import BiDirectionalRRT
# from planner.local_planner.executors.dubins_path import DubinsPathPlanner
from planner.goal_point_discover import GoalPointDiscoverResult

MIN_DIST_NONE = 0
MIN_DIST_CPU = 1
MIN_DIST_GPU = 2

MAX_PATH_SIZE=40
DIST_TO_GOAL_TOLERANCE=15
SEGMENTATION_COST = [-1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, -1]
   
class BiDirectionalRRTPlanner(LocalPathPlannerExecutor): 

    __rrt: BiDirectionalRRT
    _search: bool
    _optimize: bool
    _plan_task: Thread
    _result: PlanningResult

    def __init__(self, max_exec_time_ms: int):
        super().__init__(max_exec_time_ms)
        self.__rrt = BiDirectionalRRT(
            width=PhysicalParameters.OG_WIDTH,
            height=PhysicalParameters.OG_HEIGHT,
            perception_width_m=PhysicalParameters.OG_REAL_WIDTH,
            perception_height_m=PhysicalParameters.OG_REAL_HEIGHT,
            max_steering_angle_deg=PhysicalParameters.MAX_STEERING_ANGLE,
            vehicle_length_m=PhysicalParameters.VEHICLE_LENGTH_M,
            timeout_ms=max_exec_time_ms,
            min_dist_x=PhysicalParameters.MIN_DISTANCE_WIDTH_PX,
            min_dist_z=PhysicalParameters.MIN_DISTANCE_HEIGHT_PX,
            lower_bound_x=PhysicalParameters.EGO_LOWER_BOUND.x,
            lower_bound_z=PhysicalParameters.EGO_LOWER_BOUND.z,
            upper_bound_x=PhysicalParameters.EGO_UPPER_BOUND.x,
            upper_bound_z=PhysicalParameters.EGO_UPPER_BOUND.z,
            max_path_size_px=MAX_PATH_SIZE,
            dist_to_goal_tolerance_px=DIST_TO_GOAL_TOLERANCE,
            class_cost=SEGMENTATION_COST
        )
        self._search = False
        self._plan_task = None
        self._optimize = False


    def plan(self, planner_data: PlanningData, goal_result: GoalPointDiscoverResult) -> None:
        self._result = PlanningResult(
            planner_name="RRT*",
            ego_location=planner_data.ego_location,
            goal=planner_data.goal,
            next_goal=planner_data.next_goal,
            local_start=goal_result.start,
            local_goal=goal_result.goal,
            direction=goal_result.direction,
            path=None,
            result_type=PlannerResultType.NONE,
            timeout=False,
            total_exec_time_ms=0
        )
        self.__rrt.set_plan_data(
            img=planner_data.og.get_frame(),
            start=goal_result.start,
            goal=goal_result.goal,
            velocity_m_s=planner_data.velocity
        )
        self._search = True
        self._optimize = True
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()        

    def cancel(self) -> None:
        self._search = False
        self._optimize = False

        if self._plan_task is not None and self._plan_task.is_alive:
            self._plan_task.join()

        self._plan_task = None
        self._og = None
        
    def __update_path(self, path: list[tuple[int, int, float]]) -> None:
        if path is None:
            return
        p = [ Waypoint(x, y, h) for x,y,h in path ]
        self._result.update_path(p)
  

    def __perform_planning(self) -> None:
        
        self._search = True
        self._optimize = True
        self.set_exec_started()
        self.__rrt.search_init(MIN_DIST_GPU)
        #self.__rrt.search_init(MIN_DIST_NONE)
        
        while self._search and \
                not self.__rrt.goal_reached() and \
                not self._check_timeout() and \
                self.__rrt.loop_rrt_star(False):
                    pass
        
        self._result.set_timeout(self._check_timeout())
        self._result.set_total_exec_time_ms(self.get_execution_time())
        
        if self.__rrt.goal_reached():
            self.__update_path(self.__rrt.get_planned_path(True))
            self._result.set_result_type(PlannerResultType.VALID)
        
        self._search = False
        self._optimize = True
        
        # optimize
        while self._optimize and \
                not self._check_timeout() and \
                self.__rrt.loop_rrt_star(False):
                    pass
        
        self._result.update_path(self.__rrt.get_planned_path())
        self._result.set_total_exec_time_ms(self.get_execution_time())
        self._optimize = False

    def is_planning(self) -> bool:
        return self._search
    
    def is_optimizing(self) -> bool:
        return self._optimize

    def get_result(self) -> PlanningResult:
        return self._result
    
    def destroy(self) -> None:
        self.cancel()
        pass
    
    
