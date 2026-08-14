#include "../include/wpmp_graph.h"
#include "wpmp_data.h"

WGraph::WGraph(SearchFrame *frame)
{
    const int width = frame->width();
    const int height = frame->height();
    _node_conf = std::make_shared<Frame<int4>>(width, height);
    _node_data = std::make_unique<Frame<float4>>(width, height);
    _graph_size = _node_conf->width() * _node_conf->height();
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

