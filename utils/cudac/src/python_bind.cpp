
#include "../include/cuda_frame.h"

extern "C"
{
    void *load_frame(float *frame,
                     int width,
                     int height,
                     int min_dist_x,
                     int min_dist_z,
                     int lower_bound_x,
                     int lower_bound_z,
                     int upper_bound_x,
                     int upper_bound_z)
    {
        return new CudaFrame(frame, width, height, min_dist_x, min_dist_z, lower_bound_x, lower_bound_z, upper_bound_x, upper_bound_z);
    }

    void destroy_frame(void *self)
    {
        CudaFrame *f = (CudaFrame *)self;
        delete f;
    }

    void set_goal(void *self, int x, int z)
    {
        CudaFrame *f = (CudaFrame *)self;
        f->setGoal(x, z);
    }

    void set_goal_vectorized(void *self, int x, int z)
    {
        CudaFrame *f = (CudaFrame *)self;
        f->setGoalVectorized(x, z);
    }

    void copy_back(void *self, float *ptr)
    {
        CudaFrame *f = (CudaFrame *)self;
        f->copyBack(ptr);
    }

    void get_color_frame(void *self, u_char *ptr)
    {
        CudaFrame *f = (CudaFrame *)self;
        f->convertToColorFrame(ptr);
    }

    void compute_feasible_path(void *self, float *waypoints, int count)
    {
        CudaFrame *f = (CudaFrame *)self;
        f->checkFeasibleWaypoints(waypoints, count, true);
    }

    int get_class_cost(int segmentation_class) {
        return CudaFrame::get_class_cost(segmentation_class);
    }

    int * best_waypoint_for_heading(void *self, int goal_x, int goal_z, float heading) {
        CudaFrame *f = (CudaFrame *)self;
        return f->bestWaypointPosForHeading(goal_x, goal_z, heading);
    }

    int * best_waypoint(void *self, int goal_x, int goal_z) {
        CudaFrame *f = (CudaFrame *)self;
        return f->bestWaypointPos(goal_x, goal_z);
    }

    void free_waypoint(int* waypoint) {
        delete []waypoint;
    }


    int * best_waypoint_in_direction(void *self, int start_x, int start_z, int goal_x, int goal_z) {
        CudaFrame *f = (CudaFrame *)self;
        return f->bestWaypointInDirection(start_x, start_z, goal_x, goal_z);
    }

}
