#include "cuda_graph.h"
#include "class_def.h"
#include "kinematic_model.h"
#include <cstdlib> // for rand()
#include <ctime>   // for time()

extern void CUDA_clear(double4 *graph, double *graph_cost, int width, int height);
extern unsigned int CUDA_parallel_count(double4 *graph, unsigned int *pcount, int width, int height);
extern bool CUDA_check_connection_feasible(float3 *og, int *classCost, double *checkParams, unsigned int *pcount, double3 &start, double3 &end);
extern void __tst_CUDA_build_path(double4 *graph, float3 *og, double *checkParams, double3 &start, double3 &end, int r, int g, int b);
extern void __tst_CUDA_draw_nodes(double4 *graph, float3 *og, int width, int height, int r, int g, int b);
extern int2 CUDA_find_nearest_neighbor(double4 *graph, double *graph_cost, int3 *point, long long *bestValue, int width, int height, int x, int z);
extern int2 CUDA_find_nearest_feasible_neighbor(double4 *graph, float3 *og, int *classCost, double *checkParams, int3 *point, long long *bestValue, int x, int z);
extern void CUDA_list_elements(double4 *graph, double *graph_cost, double *result, int width, int height, int count);
extern int2 CUDA_find_best_neighbor(double4 *graph, double *graph_cost, int3 *point, long long *bestValue, int width, int height, int x, int z, float radius);
extern int2 CUDA_find_best_feasible_neighbor(double4 *graph, double *graph_cost, float3 *og, int *classCost, double *checkParams, int3 *point, long long *bestValue, int x, int z, float radius);
extern void CUDA_optimizeGraphWithNode(double4 *graph, double *graph_cost, float3 *og, int *classCost, double *checkParams, double goal_heading, int x, int z, float radius);

//#define DEBUG_DUMP

#ifdef DEBUG_DUMP

#include <opencv2/opencv.hpp>

void show_point_dump_graph(cv::Mat &mat, double x, double z, int r, int g, int b)
{
    if (x < 0 || x >= mat.cols)
        return;
    if (z < 0 || z >= mat.rows)
        return;
    cv::Vec3b &pixel = mat.at<cv::Vec3b>(static_cast<int>(z), static_cast<int>(x));
    pixel[0] = r;
    pixel[1] = g;
    pixel[2] = b;
}

void debug_dump(float3 *og, int width, int height, std::vector<double3> curve, const char *output_file)
{
    cv::Mat image(width, height, CV_8UC3);

    for (int z = 0; z < height; z++)
        for (int x = 0; x < width; x++)
        {
            int c = og[z * width + x].x;
            cv::Vec3b &pixel = image.at<cv::Vec3b>(z, x);
            pixel[0] = segmentationClassColors[c][0];
            pixel[1] = segmentationClassColors[c][1];
            pixel[2] = segmentationClassColors[c][2];
        }

    for (double3 point : curve)
    {
        show_point_dump_graph(image, point.x, point.y, 255, 255, 255);
    }

    // Save the image to verify the change
    cv::imwrite(output_file, image);
}

#endif

// TODO: receive center instead of assuming OG_WIDTH/2, OG_HEIGHT/2
CudaGraph::CudaGraph(
    int width,
    int height,
    int min_dist_x,
    int min_dist_z,
    int lower_bound_ego_x,
    int lower_bound_ego_z,
    int upper_bound_ego_x,
    int upper_bound_ego_z,
    double _rate_w,
    double _rate_h,
    double _max_steering_angle_deg,
    double _lr)
{
    std::srand(std::time(nullptr));

    this->width = width;
    this->height = height;
    this->_goal_heading_deg = 0.0;

    if (!cudaAllocMapped(&this->graph, sizeof(double4) * (width * height)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate graph with %ld bytes\n", sizeof(double4) * (width * height));
        return;
    }

    if (!cudaAllocMapped(&this->graph_cost, sizeof(double) * (width * height)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate graph_cost with %ld bytes\n", sizeof(double) * (width * height));
        cudaFreeHost(graph);
        return;
    }

    if (!cudaAllocMapped(&this->point, sizeof(int3)))
    {
        fprintf(stderr, "[CUDA RRT] unable to allocate %ld bytes for fast point output computing\n", sizeof(int2) * 2);
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        return;
    }

    classCosts = nullptr;
    const int numClasses = 29;
    if (!cudaAllocMapped(&classCosts, sizeof(int) * numClasses))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for class costs\n", sizeof(int) * numClasses);
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        return;
    }
    for (int i = 0; i < numClasses; i++)
        classCosts[i] = segmentationClassCost[i];

    if (!cudaAllocMapped(&pcount, sizeof(unsigned int)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for counting tasks\n", sizeof(unsigned int));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        return;
    }

    if (!cudaAllocMapped(&bestValue, sizeof(long long)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for computing best value tasks\n", sizeof(long long));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        cudaFreeHost(pcount);
        return;
    }

    if (!cudaAllocMapped(&checkParams, 15 * sizeof(double)))
    {
        fprintf(stderr, "[CUDA GRAPH] unable to allocate %ld bytes for the computing params\n", 30 * sizeof(double));
        cudaFreeHost(graph);
        cudaFreeHost(graph_cost);
        cudaFreeHost(point);
        cudaFreeHost(classCosts);
        cudaFreeHost(pcount);
        cudaFreeHost(bestValue);
        return;
    }

    checkParams[0] = width;
    checkParams[1] = height;
    checkParams[2] = min_dist_x / 2;
    checkParams[3] = min_dist_z / 2;
    checkParams[4] = lower_bound_ego_x;
    checkParams[5] = lower_bound_ego_z;
    checkParams[6] = upper_bound_ego_x;
    checkParams[7] = upper_bound_ego_z;
    checkParams[8] = _rate_w;
    checkParams[9] = _rate_h;
    checkParams[10] = _max_steering_angle_deg;
    checkParams[11] = _lr;
    checkParams[12] = 1.0;
    checkParams[13] = width / 2;  // OG center
    checkParams[14] = height / 2; // OG center

    _center.x = checkParams[13];
    _center.y = checkParams[14];
    _center.z = 0.0;

    // clear();
}

CudaGraph::~CudaGraph()
{
    cudaFreeHost(this->graph);
    cudaFreeHost(this->graph_cost);
    cudaFreeHost(this->point);
    cudaFreeHost(this->classCosts);
    cudaFreeHost(this->pcount);
    cudaFreeHost(this->bestValue);
    cudaFreeHost(this->checkParams);
}

void CudaGraph::clear()
{
    CUDA_clear(this->graph, this->graph_cost, this->width, this->height);
    _unordered_nodes.clear();
}

void CudaGraph::setVelocity(double velocity_meters_per_s)
{
    checkParams[12] = velocity_meters_per_s;
}

void CudaGraph::add(int x, int z, double heading, int parent_x, int parent_z, double cost)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;

    this->graph[pos].x = parent_x;
    this->graph[pos].y = parent_z;
    this->graph[pos].z = heading;
    this->graph[pos].w = 1.0;
    this->graph_cost[pos] = cost;
    _unordered_nodes.push_back(int2{x, z});
}

double CudaGraph::getHeading(int x, int z)
{
    if (x > width || x < 0)
        return 0.0;
    if (z > height || z < 0)
        return 0.0;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return 0.0;

    if (this->graph[pos].w != 1.0)
        return 0.0;

    return graph[pos].z;
}

void CudaGraph::remove(int x, int z)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;

    if (this->graph[pos].w == 1.0)
    {
        for (int i = 0; i < _unordered_nodes.size(); i++)
        {
            if (_unordered_nodes[i].x != x || _unordered_nodes[i].y != z)
                continue;
            if (i != _unordered_nodes.size() - 1)
            {
                _unordered_nodes[i] = _unordered_nodes[_unordered_nodes.size() - 1];
                _unordered_nodes.pop_back();
            }
        }
    }

    this->graph[pos].w = 0.0;
}

unsigned int CudaGraph::count()
{
    // return CUDA_parallel_count(graph, pcount, width, height);
    return _unordered_nodes.size();
}

bool CudaGraph::checkInGraph(int x, int z)
{
    if (x > width || x < 0)
        return false;
    if (z > height || z < 0)
        return false;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return false;

    return this->graph[pos].w == 1.0;
}

void CudaGraph::setParent(int x, int z, int parent_x, int parent_z)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return;
    if (this->graph[pos].w == 0.0)
        return;

    this->graph[pos].x = parent_x;
    this->graph[pos].y = parent_z;
}
double3 CudaGraph::getParent(int x, int z)
{
    double3 p;
    p.x = -1;
    p.y = -1;
    p.z = -1;

    if (x > width || x < 0)
        return p;
    if (z > height || z < 0)
        return p;

    int pos = z * width + x;

    if (pos > width * height || pos < 0)
        return p;
    if (this->graph[pos].w == 0.0)
        return p;

    p.x = this->graph[pos].x;
    p.y = this->graph[pos].y;
    p.z = 0.0;

    if (p.x < 0 || p.z < 0)
        return p;

    int pos_parent = p.y * width + p.x;

    if (pos_parent >= width * height || pos_parent < 0)
        return p;

    p.z = this->graph[pos_parent].z;
    return p;
}

void CudaGraph::setCost(int x, int z, double cost)
{
    if (x > width || x < 0)
        return;
    if (z > height || z < 0)
        return;
    int pos = z * width + x;
    if (pos > width * height || pos < 0)
        return;
    if (this->graph[pos].w == 0.0)
        return;

    this->graph_cost[pos] = cost;
}
double CudaGraph::getCost(int x, int z)
{
    if (x > width || x < 0)
        return -1;
    if (z > height || z < 0)
        return -1;
    int pos = z * width + x;
    if (pos > width * height || pos < 0)
        return -1;
    if (this->graph[pos].w == 0.0)
        return -1;

    return this->graph_cost[pos];
}

// void CudaGraph::list(double *result, int count)
// {
//     CUDA_list_elements(graph, graph_cost, result, width, height, count);
// }

bool CudaGraph::checkConnectionFeasible(float3 *og, double3 &start, double3 end)
{
    return CUDA_check_connection_feasible(og, classCosts, checkParams, pcount, start, end);
}

void CudaGraph::drawNodes(float3 *og)
{
    __tst_CUDA_draw_nodes(graph, og, width, height, 0, 0, 255);
}

void CudaGraph::drawKinematicPath(float3 *og, double3 &start, double3 &end)
{
    __tst_CUDA_build_path(graph, og, checkParams, start, end, 255, 255, 255);
}

int2 CudaGraph::find_nearest_neighbor(int x, int z)
{
    return CUDA_find_nearest_neighbor(graph, graph_cost, point, bestValue, width, height, x, z);
}

int2 CudaGraph::find_nearest_feasible_neighbor(float3 *og, int x, int z)
{
    return CUDA_find_nearest_feasible_neighbor(graph, og, classCosts, checkParams, point, bestValue, x, z);
}

int2 CudaGraph::find_best_neighbor(int x, int z, float radius)
{
    return CUDA_find_best_neighbor(graph, graph_cost, point, bestValue, width, height, x, z, radius);
}

int2 CudaGraph::find_best_feasible_neighbor(float3 *og, int x, int z, float radius)
{
    return CUDA_find_best_feasible_neighbor(graph, graph_cost, og, classCosts, checkParams, point, bestValue, x, z, radius);
}

// orders nodes within a radius to verify if they should use x,z as their parent node.
void CudaGraph::optimizeGraphWithNode(float3 *og, int x, int z, float radius)
{
    return CUDA_optimizeGraphWithNode(graph, graph_cost, og, classCosts, checkParams, _goal_heading_deg, x, z, radius);
}

void CudaGraph::setGoalHeading(double heading_deg)
{
    _goal_heading_deg = heading_deg;
}

bool CudaGraph::connectToGraph(float3 *og, int parent_x, int parent_z, int x, int z)
{
    if (!checkInGraph(parent_x, parent_z))
        return false;

    if (checkInGraph(x, z))
        return true;

    double3 start, end;

    start.x = parent_x;
    start.y = parent_z;
    start.z = getHeading(parent_x, parent_z);

    end.x = x;
    end.y = z;
    end.z = 0; // not used

    std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(_center,
                                                                     checkParams[8],
                                                                     checkParams[9],
                                                                     checkParams[11],
                                                                     checkParams[10],
                                                                     start,
                                                                     end,
                                                                     checkParams[12],
                                                                     false);

    double last_heading = 0.0;
    double last_x;
    double last_z;

    for (double3 p : curve)
    {
        last_x = p.x;
        last_z = p.y;
        last_heading = p.z;

        if (!ConstraintsCheckCPU::computeFeasibleForAngle(og,
                                                          classCosts,
                                                          p.x,
                                                          p.y,
                                                          p.z,
                                                          checkParams[0],
                                                          checkParams[1],
                                                          checkParams[2],
                                                          checkParams[3],
                                                          checkParams[4],
                                                          checkParams[5],
                                                          checkParams[6],
                                                          checkParams[7]))
            return false;
    }

    last_heading = CurveGenerator::to_degrees(end.z);
    double cost = getCost(parent_x, parent_z) + CurveGenerator::compute_node_diff_cost(start, end, _goal_heading_deg, last_heading);

    add(static_cast<int>(last_x),
        static_cast<int>(last_z),
        last_heading,
        parent_x,
        parent_z,
        cost);

    return true;
}

int2 CudaGraph::deriveNode(float3 *og, int parent_x, int parent_z, double angle_deg, double size)
{
    int2 res;
    res.x = -1;
    res.y = -1;

    if (!checkInGraph(parent_x, parent_z))
        return res;

    double3 start;

    start.x = parent_x;
    start.y = parent_z;
    start.z = getHeading(parent_x, parent_z);

    std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(_center,
                                                                     checkParams[8],
                                                                     checkParams[9],
                                                                     checkParams[11],
                                                                     checkParams[10],
                                                                     start,
                                                                     checkParams[12],
                                                                     angle_deg,
                                                                     size,
                                                                     false);
    double3 last = curve[curve.size() - 1];
    //printf("deriving %f, %f to %f, %f \n", start.x, start.y, last.x, last.y);

    double3 end;
    end.x = parent_x;
    end.y = parent_z;
    end.z = start.z;

#ifdef DEBUG_DUMP
    debug_dump(og, width, height, curve, "curve_debug_try.png");
#endif

    for (double3 p : curve)
    {
        end.x = p.x;
        end.y = p.y;
        end.z = p.z;

        if (!ConstraintsCheckCPU::computeFeasibleForAngle(og,
                                                          classCosts,
                                                          p.x,
                                                          p.y,
                                                          p.z,
                                                          checkParams[0],
                                                          checkParams[1],
                                                          checkParams[2],
                                                          checkParams[3],
                                                          checkParams[4],
                                                          checkParams[5],
                                                          checkParams[6],
                                                          checkParams[7]))
            return res;
    }

    double last_heading = CurveGenerator::to_degrees(end.z);
    double cost = getCost(parent_x, parent_z) + CurveGenerator::compute_node_diff_cost(start, end, _goal_heading_deg, last_heading);

    res.x = static_cast<int>(end.x);
    res.y = static_cast<int>(end.y);

    if (checkInGraph(res.x, res.y))
    {
        res.x = -1;
        res.y = -1;
        return res;
    }

    add(res.x,
        res.y,
        last_heading,
        parent_x,
        parent_z,
        cost);

    return res;
}
int CudaGraph::__random_gen(int min, int max)
{
    return min + std::rand() % (max - min);
}

int2 CudaGraph::get_random_node()
{
    int i = __random_gen(0, _unordered_nodes.size());
    return _unordered_nodes[i];
}

std::vector<int2>& CudaGraph::list()
{
    return _unordered_nodes;
}