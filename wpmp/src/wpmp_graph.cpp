#include "../include/wpmp_graph.h"
#include "wpmp_data.h"

WGraph::WGraph(int width, int height)
{
    _node_conf = std::make_shared<Frame<int4>>(width, height);
    _node_data = std::make_unique<Frame<float4>>(width, height);
    _graph_size = _node_conf->width() * _node_conf->height();
    _class_costs = nullptr;

    #ifdef DRIVELESS_CUDA_ENABLED
    _search_space_params = std::make_unique<CudaPtr<int>>(20);
    _search_space_params->get()[FRAME_PARAM_WIDTH] = width;
    _search_space_params->get()[FRAME_PARAM_HEIGHT] = height;
#else
    _search_space_params = std::make_unique<int[]>(20);
    _search_space_params.get()[FRAME_PARAM_WIDTH] = width;
    _search_space_params.get()[FRAME_PARAM_HEIGHT] = height;
#endif
}

void WGraph::clear()
{
    _node_conf->clear();
}

void WGraph::set_start(int x, int z, float heading)
{
    int pos = COMPUTE_POS(_node_conf->width(), x, z);
    SET_NODE_PARENT(_node_conf->getPtr(), pos, -1, -1);
    SET_NODE_HEADING(_node_conf->getPtr(), pos, heading);
}

void WGraph::set_physical_params(
    float perception_width_in_meters,
    float perception_height_in_meters,
    angle max_steering_angle,
    float wheelbase)
{
    _perception_dim_in_meters = {perception_width_in_meters, perception_height_in_meters};
    _max_steering_angle_rad = max_steering_angle.rad();
    _wheelbase = wheelbase;
}

void WGraph::set_search_params(
    int2 min_distance_in_px,
    int2 lower_bound,
    int2 upper_bound)
{
#ifdef DRIVELESS_CUDA_ENABLED
    _search_space_params->get()[FRAME_PARAM_MIN_DIST_X] = TO_INT((float)min_distance_in_px.x / 2);
    _search_space_params->get()[FRAME_PARAM_MIN_DIST_Z] = TO_INT((float)min_distance_in_px.y / 2);
    _search_space_params->get()[FRAME_PARAM_LOWER_BOUND_X] = lower_bound.x;
    _search_space_params->get()[FRAME_PARAM_LOWER_BOUND_Z] = lower_bound.y;
    _search_space_params->get()[FRAME_PARAM_UPPER_BOUND_X] = upper_bound.x;
    _search_space_params->get()[FRAME_PARAM_UPPER_BOUND_Z] = upper_bound.y;
#else
    _search_space_params.get()[FRAME_PARAM_MIN_DIST_X] = TO_INT((float)min_distance_in_px.x / 2);
    _search_space_params.get()[FRAME_PARAM_MIN_DIST_Z] = TO_INT((float)min_distance_in_px.y / 2);
    _search_space_params.get()[FRAME_PARAM_LOWER_BOUND_X] = lower_bound.x;
    _search_space_params.get()[FRAME_PARAM_LOWER_BOUND_Z] = lower_bound.y;
    _search_space_params.get()[FRAME_PARAM_UPPER_BOUND_X] = upper_bound.x;
    _search_space_params.get()[FRAME_PARAM_UPPER_BOUND_Z] = upper_bound.y;
#endif
}

void WGraph::set_frame_class_costs(std::vector<float> costs)
{
#ifdef DRIVELESS_CUDA_ENABLED
    _class_costs = std::make_unique<CudaPtr<float>>(costs.size());
    int i = 0;
    for (float p : costs)
    {
        _class_costs->get()[i++] = p;
    }
#else
    _class_costs = std::make_unique<float[]>(costs.size());
    int i = 0;
    for (float p : costs)
    {
        _class_costs.get()[i++] = p;
    }
#endif
}
