#include "../include/cuda_basic.h"
#include "../include/cpu_parallel_processor.h"
#include "../include/search_frame.h"

#define NUM_POINTS_ON_MEAN 5

float __computeHeading(int p1_x, int p1_y, int p2_x, int p2_y, bool *valid, int width, int height)
{
    *valid = false;
    if (p1_x == p2_x && p1_y == p2_y)
        return 0.0;

    if (p1_x < 0 || p1_y < 0 || p2_x < 0 || p2_y < 0)
        return 0.0;

    if (p1_x >= width || p1_y >= height || p2_x >= width || p2_y >= height)
        return 0.0;

    float dx = p2_x - p1_x;
    float dz = p2_y - p1_y;
    *valid = true;
    float heading = CUDART_PI_F / 2 - atan2f(-dz, dx);

    if (heading > CUDART_PI_F) // greater than 180 deg
        heading = heading - 2 * CUDART_PI_F;

    return heading;
}

float ___computeMeanHeading(std::vector<Waypoint> &path, int pos, bool *valid, int width, int height)
{
    float heading = 0.0;
    int count = 0;
    int size = path.size();

    for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
    {
        bool v = false;

        if (pos + j >= size)
            break;

        heading += __computeHeading(path[pos].x(), path[pos].z(), path[pos + j].x(), path[pos + j].z(), &v, width, height);

        if (!v)
            break;

        count++;
    }

    if (count != NUM_POINTS_ON_MEAN)
    {
        count = 0;
        // compute in reverse
        for (int j = 1; j <= NUM_POINTS_ON_MEAN; j++)
        {
            bool v = false;
            if (pos - j < 0)
            {
                *valid = false;
                return 0.0;
            }
            heading += __computeHeading(path[pos - j].x(), path[pos - j].z(), path[pos].x(), path[pos].z(), &v, width, height);
            if (!v)
                break;
            count++;
        }
    }

    *valid = count > 0;

    if (*valid)
        return heading / count;

    return 0.0;
}

class ParallelComputeMeanHeading : public ParallelProcessor
{
    std::vector<Waypoint> _path;
    const int _width;
    const int _height;
    const int _numThreadHandlers;
    bool _computeValid;

public:
    ParallelComputeMeanHeading(std::vector<Waypoint> path,
                               int width, int height, int numThreadHandlers) : ParallelProcessor(numThreadHandlers, 1, path.size()),
                                                                               _path(path), _width(width), _height(height), _numThreadHandlers(numThreadHandlers), _computeValid(true) {}

    void handler(int threadId) override
    {
        if (threadId >= _path.size())
            return;

        bool valid = false;
        float heading = ___computeMeanHeading(_path, threadId, &valid, _width, _height);
        if (!valid)
            _computeValid = false;
        _path[threadId].set_heading(heading);
    }

    bool isValid() {
        return _computeValid;
    }
};

bool SearchFrame::computePathHeadings(std::vector<Waypoint> path)
{
    ParallelComputeMeanHeading comp(path, width(), height(), _numCPUThreadHandlers);
    comp.runAndWait();
    return comp.isValid();
}
