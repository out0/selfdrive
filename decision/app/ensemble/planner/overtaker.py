
from pydriveless import Waypoint, angle
from pydriveless import SearchFrame, SearchParams, EgoParams
from .. model.planner_executor import LocalPlannerExecutor
from .. model.planning_result import PlannerResultType
from .. model.planning_data import PlanningData
from .. model.physical_paramaters import PhysicalParameters
from .dubins_path import Dubins
#from .debug import Debug
import math


class Overtaker(LocalPlannerExecutor):
    __dubins: Dubins
    _search_params: SearchParams

    DEBUG=False

    def __init__(self, ego_params: EgoParams):
        super().__init__("Overtaker")
        a = ego_params.max_steering_angle.deg()
        self.__dubins = Dubins(a, 0.1*a)
        start = None
   
    def _planning_init(self, search_params: SearchParams) -> bool:
        self._search_params = search_params
        self.set_timeout(search_params.timeout_ms)
        return True


    def _loop_plan(self) -> bool:
        start =  self._search_params.start
        goal = self._search_params.goal
        frame = self._search_params.frame

        path = self.__try_direct_path(frame, self._search_params.min_distance, start, goal)

        if path is not None:
            self._set_planning_result(PlannerResultType.VALID, path)
            return False

        if self._check_timeout(): 
            return False

        try_left_first = goal.x < start.x        
        
        if try_left_first:
            path = self.__relocate_left(frame, start, goal, self._search_params.min_distance)

            if path is None:
                if self._check_timeout(): 
                    return False                
                path = self.__relocate_right(frame, start, goal, self._search_params.min_distance)

            if path is not None:
                self._set_planning_result(PlannerResultType.VALID, path)           
            
        else:
            path = self.__relocate_right(frame, start, goal, self._search_params.min_distance)

            if path is None:
                if self._check_timeout(): 
                    return False
                path = self.__relocate_left(frame, start, goal, self._search_params.min_distance)

            if path is not None:
                self._set_planning_result(PlannerResultType.VALID, path) 

        return False                
        
    def _loop_optimize(self) -> bool:
        # ignore
        return False
    
    def __try_direct_path(self, og: SearchFrame, min_distance: tuple[int, int], start: Waypoint, goal: Waypoint):
        path = self.__build_overtake_path(og, start, goal)
        #Debug.log_path(path, "direct.log")

        if not self.__check_path_feasible(og, min_distance, path):
            return None
        return path
    
    def __check_path_feasible(self, og: SearchFrame, min_distance: tuple[int, int], path: list[Waypoint]) -> bool:
        if path is None or len(path) <= 2:
            return False
        
        return og.check_feasible_path(min_distance=min_distance, path=path)
    


    def __build_overtake_path(self, og: SearchFrame, start: Waypoint, goal: Waypoint) -> list[Waypoint]:
        return self.__dubins.build_path(start, goal, og.width() - 1, og.height() - 1)   
    def __find_first_feasible_goal(self, og: SearchFrame, z: int, x_init: int, x_limit: int, heading: angle) -> int:
        inc = 1
        if x_init > x_limit:
            inc = -1
        
        for i in range(x_init, x_limit, inc):
            if og.is_traversable(i, z, heading, precision_check=True):
                return i
        return -1
    def __relocate_left(self, og: SearchFrame, start: Waypoint, goal: Waypoint, min_distance: tuple[int, int]) -> list[Waypoint]:
        if self._check_timeout(): return None
        x_min = self.__find_first_feasible_goal(og, goal.z, x_init=0, x_limit=goal.x, heading=goal.heading)
        if x_min < 0: return None
        
        x = math.floor(0.5 * (goal.x + x_min))
        
        if x < x_min or x > goal.x:
            return None

        new_goal = Waypoint(x, goal.z, goal.heading)
        path = self.__build_overtake_path(og, start, new_goal)
            
        if og.check_feasible_path(min_distance, path):
            return path
        
        return self.__relocate_step(og, start, new_goal, min_distance, step=-2, x_min=x_min, x_max=og.width())
    def __relocate_right(self, og: SearchFrame, start: Waypoint, goal: Waypoint, min_distance: tuple[int, int]) -> list[Waypoint]:
        if self._check_timeout(): return None
        x_max = self.__find_first_feasible_goal(og, goal.z, x_init=goal.x, x_limit=og.width(), heading=goal.heading)        
        if x_max < 0: return None
        
        x = math.floor(0.5 * (goal.x + x_max))
        
        if x < goal.x or x > x_max:
            return False

        new_goal = Waypoint(x, goal.z, goal.heading)
        path = self.__build_overtake_path(og, start, new_goal)
        
        if og.check_feasible_path(min_distance, path):
            return path
        
        return self.__relocate_step(og, start, new_goal, min_distance, step=2, x_min=x, x_max=x_max)
    def __relocate_step(self, og: SearchFrame, start: Waypoint, goal: Waypoint,  min_distance: tuple[int, int], step: int, x_min: int, x_max: int) -> list[Waypoint]:

        new_x = goal.x + step

        if goal.x <= x_min or goal.x >= x_max:
            return None

        new_goal = Waypoint(new_x, goal.z, goal.heading)
        path = self.__build_overtake_path(og, start, new_goal)
        
        if og.check_feasible_path(min_distance, path):
            return path
        
        return self.__relocate_step(og, start, new_goal, min_distance, step=step, x_min=x_min, x_max=x_max)
    
    
    
   
