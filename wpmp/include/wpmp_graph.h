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

    
#ifdef DRIVELESS_CUDA_ENABLED
    std::unique_ptr<CudaPtr<float>> _class_costs;
    std::unique_ptr<CudaPtr<int>> _search_space_params;
#else
    std::unique_ptr<float[]> _class_costs;
    std::unique_ptr<int[]> _search_space_params;
    int _bestValue;
#endif


    long _graph_size;
    float2 _perception_dim_in_meters;
    float _max_steering_angle_rad;
    float _wheelbase;

public:
    WGraph(int width, int height);

    void clear();

    void set_start(int x, int z, float heading);

    void set_physical_params(
        float perception_width_in_meters,
        float perception_height_in_meters,
        angle max_steering_angle,
        float wheelbase);

    void set_search_params(
        int2 min_distance_in_px,
        int2 lower_bound,
        int2 upper_bound);

    void set_frame_class_costs(std::vector<float> costs);

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