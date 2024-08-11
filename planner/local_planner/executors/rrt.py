from model.waypoint import Waypoint
from planner.local_planner.local_planner_executor import LocalPathPlannerExecutor
from planner.local_planner.local_planner import PlanningData, PlanningResult, PlannerResultType
from threading import Thread
import math, numpy as np
from planner.local_planner.executors.dubins_curves import Dubins
import random, sys

class RRTNode:
    parent: 'RRTNode'
    x: int
    z: int
    cost: float
    
    def __init__(self, x: int, z: int, parent: 'RRTNode') -> None:
        self.x = x
        self.z = z
        self.parent = parent
        self.cost = 0
        
    def compute_node_dist(n1: 'RRTNode', x: int, z: int) -> float:
        dx = x - n1.x
        dz = z - n1.z
        return math.sqrt(dx ** 2 + dz ** 2)


class RRTPlanner (LocalPathPlannerExecutor):
    _search: bool
    _plan_task: Thread
    _frame: np.ndarray
    _result: PlanningResult
    _plnning_data: PlanningData
    _post_plan_smooth: bool
    _width: int
    _height: int
    _max_step: float
    _node_list: list[RRTNode]
    
    NAME = "RRT"
    
    def __init__(self,  
                 max_exec_time_ms: int, 
                 max_steps: float
                 ) -> None:
        super().__init__(max_exec_time_ms)
        self._search = False
        self._plan_task = None
        self._result = None
        self._max_step = max_steps
        
        
    def plan(self, planner_data: PlanningData, partial_result: PlanningResult) -> None:
        
        self._frame = planner_data.og.get_frame()
        self._width = planner_data.og.width()
        self._height = planner_data.og.height()
        self._node_list = []
        self._search = True
        self._result = partial_result
        self._plan_task = Thread(target=self.__perform_planning)
        self._plan_task.start()

    def cancel(self) -> None:
        self._search = False

        if self._plan_task is not None and self._plan_task.is_alive:
            self._plan_task.join()

        self._plan_task = None
        self._og = None

    def is_planning(self) -> bool:
        return self._search
       

    def get_result(self) -> PlanningResult:
        return self._result

    
        
    def __generate_random_point(self) -> tuple[int, int]:
        w = random.randint(0, self._width)
        h = random.randint(0, self._height)
        return (w, h)
    
    def __find_nearest_node(self, x: int, z: int) -> RRTNode:
        best_dist = sys.maxsize
        nearest = None
        for n in self._node_list:
            dist = RRTNode.compute_node_dist(n, x, z)
            if dist < best_dist:
                best_dist = dist
                nearest = n
        return nearest
    
    
    def __generate_node_towards_point(self, base_point: RRTNode, x_rand: int, z_rand: int) -> RRTNode:
        dist = RRTNode.compute_node_dist(base_point, x_rand, z_rand)
        
        if dist < self._max_step:
            Dubins.dubins_path()
            return RRTNode(x_rand, z_rand, None)
        
        ## TODO: Add Dubins here
        slope = math.atan2(z_rand - base_point.z, x_rand - base_point.x)
        x = base_point.x + math.floor(self._max_step * math.cos(slope))
        z = base_point.z + math.floor(self._max_step * math.sin(slope))
        return RRTNode(x, z, None)
    
    def __check_path_feasible(self, p1: RRTNode, p2: RRTNode) -> bool:
        dx = abs(p2.x - p1.x)
        dz = abs(p2.z - p1.z)
        steps = max(dz, dx)
        
        if steps == 0:
            return False
        
        x_inc = dx / steps
        z_inc = dz / steps
        
        x = p1.x
        z = p1.z
        
        for _ in range(steps):
            x += x_inc
            z += z_inc
            if not self.__check_feasible(math.floor(x), math.floor(z)):
                return False
        
        return True
    
    def __perform_planning(self) -> None:
        self.set_exec_started()
        self._node_list.append(RRTNode(self._result.local_start.x, self._result.local_start.z, None))
        
        loop_search = self._search
        self._rst_timeout()
        
        while loop_search:
            if self._check_timeout():
                loop_search = False
                continue
            
            x_rand, z_rand = self.__generate_random_point()
            
            nearest = self.__find_nearest_node(x_rand, z_rand)
            
            new_node: RRTNode = self.__generate_node_towards_point(nearest, x_rand, z_rand)
            
            if not self.__check_path_feasible(nearest, new_node):
                continue
            
            near_nodes: list[RRTNode] = []
            for n in self._node_list:
                if RRTNode.compute_node_dist(n, new_node.x, new_node.z) < self._max_step:
                    near_nodes.append(n)
            
            best_cost_parent = nearest
            best_cost = RRTNode.compute_node_dist(nearest, new_node.x, new_node.z) + best_cost_parent.cost
            
            for n in near_nodes:
                cost = RRTNode.compute_node_dist(new_node, n.x, n.z)
                if cost < best_cost and self.__check_path_feasible(n, new_node):
                    best_cost = cost
                    best_cost_parent = n
            
            new_node.parent = best_cost_parent
            new_node.cost = best_cost
            
            self._node_list.append(new_node)
        
        # finding the path
        
        path: list[Waypoint] = []
        current = self.__find_nearest_node(self._result.local_goal.x, self._result.local_goal.z)
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        self._result.path = path
        
        if len(path) <= 3:
            self._result.result_type = PlannerResultType.INVALID_PATH
        else:
            self._result.result_type = PlannerResultType.VALID
        
        self._result.total_exec_time_ms = self.get_execution_time()
        self._search = False
