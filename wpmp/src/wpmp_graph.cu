#include "../include/wpmp_graph.h"
#include <driveless/search_zone_utils.h>
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


