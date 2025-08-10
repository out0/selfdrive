#include "../../include/search_frame.h"
#include <tuple>

extern "C"
{
    void *search_frame_initialize(int width, int height, int lowerBoundX, int lowerBoundZ, int upperBoundX, int upperBoundZ)
    {
        return new SearchFrame(width, height, {lowerBoundX, lowerBoundZ}, {upperBoundX, upperBoundZ});
    }

    void search_frame_destroy(void *self)
    {
        SearchFrame *frame = (SearchFrame *)self;
        delete frame;
    }

    void search_frame_copy_data(void *self, float *ptr)
    {
        SearchFrame *frame = (SearchFrame *)self;
        frame->copyFrom(ptr);
    }

    void search_frame_copy_back(void *self, float *ptr)
    {
        SearchFrame *frame = (SearchFrame *)self;
        frame->copyTo(ptr);
    }

    void export_to_color_frame(void *self, uchar *dest)
    {
        SearchFrame *frame = (SearchFrame *)self;
        frame->exportToColorFrame(dest);
    }

    void set_class_colors(void *self, int numClasses, uint *colors)
    {
        SearchFrame *frame = (SearchFrame *)self;
        std::vector<std::tuple<int, int, int>> classColors;
        classColors.reserve(numClasses);

        for (int i = 0; i < numClasses; i++)
        {
            int pos = 3 * i;
            classColors.push_back({colors[pos], colors[pos + 1], colors[pos + 2]});
        }

        frame->setClassColors(classColors);
    }

    void set_class_costs(void *self, int numClasses, float *costs)
    {
        SearchFrame *frame = (SearchFrame *)self;
        std::vector<float> classCosts;
        classCosts.reserve(numClasses);

        for (int i = 0; i < numClasses; i++)
        {
            classCosts.push_back(costs[i]);
        }
        frame->setClassCosts(classCosts);
    }

    float get_class_cost(void *self, int classId)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->getClassCost(classId);
    }

    bool is_obstacle(void *self, int x, int z)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->isObstacle(x, z);
    }

    /// @brief Returns the pre-computed (using setGoal()) cost for pos x,z, given by its distance to the goal, multiplied by the class cost provided by function setClassCosts()
    /// @param x
    /// @param z
    /// @return
    double get_cost(void *self, int x, int z)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->getCost(x, z);
    }

    /// @brief Sets a goal point to pre-compute costs and traversability for every x,z position in the search space.
    /// @param x
    /// @param z
    void process_safe_distance_zone(void *self, bool compute_vectorized, int minDistX, int minDistZ)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->processSafeDistanceZone({minDistX, minDistZ}, compute_vectorized);
    }

    /// @brief Returns a vector with true for each feasible and false for each non-feasible waypoint in the path
    /// @param path
    /// @param computeHeadings should compute headings when calculating feasibility
    /// @return
    bool check_feasible_path(void *self, float *path, int count, int minDistX, int minDistZ, bool feasibleInfoForAllPoints)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->checkFeasiblePath(path, count, minDistX, minDistZ, feasibleInfoForAllPoints);
    }

    int get_traversability(void *self, int x, int z)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return static_cast<int>((*frame)[{x, z}].z);
    }

    bool is_traversable(void *self, int x, int z)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->isTraversable(x, z);
    }

    bool is_traversable_on_angle(void *self, int x, int z, float angle_rad, bool precision_check)
    {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->isTraversable(x, z, angle::rad(angle_rad), precision_check);
    }

    void read_cell(void *self, int x, int z, float *res)
    {
        SearchFrame *frame = (SearchFrame *)self;
        float3 p = (*frame)[{x, z}];
        res[0] = p.x;
        res[1] = p.y;
        res[2] = p.z;
    }
    void write_cell(void *self, int x, int z, float v1, float v2, float v3)
    {
        SearchFrame *frame = (SearchFrame *)self;
        (*frame)[{x, z}].x = v1;
        (*frame)[{x, z}].y = v2;
        (*frame)[{x, z}].z = v3;
        //frame->setValues(x, z, v1, v2, v3);
    }


    void process_distance_to_goal(void *self, int x, int z) {
        SearchFrame *frame = (SearchFrame *)self;
        frame->processDistanceToGoal(x, z);
    }

    float get_distance_to_goal(void *self, int x, int z) {
        SearchFrame *frame = (SearchFrame *)self;
        return frame->getDistanceToGoal(x, z);
    }

}
