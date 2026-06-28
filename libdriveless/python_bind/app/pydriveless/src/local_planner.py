from abc import ABC, abstractmethod
from .waypoint import Waypoint


class LocalPlanner(ABC):
    @abstractmethod
    def initialize(self, copy_intrinsic_costs_from_frame: bool = False) -> None:
        """Initializes the local planner.

        Args:
            copy_intrinsic_costs_from_frame: Copies the values in the frame's channel G as intrinsic values to support using cost maps.
        """
        raise NotImplementedError

    @abstractmethod
    def planning_loop(self) -> bool:
        """Executes a planning loop.

        Returns:
            False if the planner should stop planning.
        """
        raise NotImplementedError

    @abstractmethod
    def path_optimize_loop(self) -> bool:
        """Executes an optimization loop.

        Returns:
            False if the planner should stop optimizing.
        """
        raise NotImplementedError

    @abstractmethod
    def goal_reached(self) -> bool:
        """Checks if the planner reached the goal.

        Returns:
            True in case of goal reached.
        """
        raise NotImplementedError

    @abstractmethod
    def get_planned_path(self) -> tuple[list[Waypoint], float]:
        """Returns the planned path and the path cost.

        Returns:
            A tuple with a vector of waypoints and a float representing the total path cost.
        """
        raise NotImplementedError

    @abstractmethod
    def get_interpolated_planned_path(self) -> tuple[list[Waypoint], float]:
        """Returns an interpolation of the planned path and the original path cost.

        Returns:
            A tuple with a vector of waypoints and a float representing the total path cost.
        """
        raise NotImplementedError