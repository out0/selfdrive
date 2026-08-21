#pragma once

#ifndef __WPMP_GRAPH_H
#define __WPMP_GRAPH_H

#include <driveless/angle.h>
#include <driveless/waypoint.h>
#include <driveless/frame.h>
#include <driveless/search_params.h>

class WGraph
{
private:
    std::shared_ptr<Frame<int4>> _node_conf;
    std::shared_ptr<Frame<float4>> _node_data;
    long _graph_size;

public:
    WGraph(SearchFrame *frame);

    void clear();

    void set_start(int x, int z, float heading);

    void compute_goal_wave(SearchFrame *frame, Waypoint &goal);

    inline std::shared_ptr<Frame<int4>> get_node_conf()
    {
        return _node_conf;
    }
    inline std::shared_ptr<Frame<float4>> get_node_data()
    {
        return _node_data;
    }
};

#endif