#pragma once

#ifndef __WPMP_GRAPH_H
#define __WPMP_GRAPH_H
#include <driveless/angle.h>
#include <driveless/waypoint.h>
#include <driveless/frame.h>
#include <driveless/search_params.h>
#include <driveless/cuda_basic.h>
#include <queue>

#ifndef DRIVELESS_CUDA_ENABLED
#include <atomic>
#endif



class WGraph
{
private:
    std::shared_ptr<Frame<int4>> _node_conf;
    std::shared_ptr<Frame<float4>> _node_data;
    long _graph_size;
    std::priority_queue<float, std::vector<float>, std::greater<float>>
        min_queue;

#ifdef DRIVELESS_CUDA_ENABLED
    CudaPtr<long long> _best_cost;
#else
    std::atomic<long long> _best_cost;
#endif

public:
    WGraph(SearchFrame *frame);

    void clear();

    void set_start(int x, int z, float heading);

    void compute_goal_wave(SearchFrame *frame, Waypoint &goal);

    bool expand(SearchFrame *frame, int x, int z, int max_size_px, Waypoint &goal, float max_error_dist_to_goal_px, float max_heading_error_rad);

    bool connect_to_goal(Waypoint &goal);

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