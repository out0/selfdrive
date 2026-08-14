#include "../wpmp_data.h"
#include "../../include/wpmp_graph.h"
#include <driveless/cpu_parallel_processor.h>

extern __device__ __host__ void to_goal_wave(float3 *frame,
                                      int *params,
                                      float *class_costs,
                                      int pos,
                                      float3 goal,
                                      float wheelbase,
                                      float delta_max_rad,
                                      int4 *node_conf,
                                      float4 *node_data,
                                      uint4 *search_zone_info);

extern __device__ __host__ void to_goal_wave_2(float3 *frame,
                                        int *params,
                                        float *class_costs,
                                        int pos,
                                        float3 goal,
                                        float wheelbase,
                                        float delta_max_rad,
                                        int4 *node_conf,
                                        float4 *node_data,
                                        uint4 *search_zone_info);                                         

class GoalWaveParallelCompute : public ParallelProcessor
{
    int _max;
    float3 *_frame;
    int *_params;
    float *_class_costs;
    int _pos;
    float3 _goal;
    float _wheelbase;
    float _delta_max_rad;
    int4 *_node_conf;
    float4 *_node_data;
    uint4 *_search_zone_info;

protected:
    void handler(int threadId) override
    {
        if (threadId >= _max)
            return;

        to_goal_wave(_frame, _params, _class_costs, threadId, _goal, _wheelbase, _delta_max_rad, _node_conf, _node_data, _search_zone_info);
    }

public:
    GoalWaveParallelCompute(
        float3 *frame,
        int *params,
        float *class_costs,
        float3 goal,
        float wheelbase,
        float delta_max_rad,
        int4 *node_conf,
        float4 *node_data,
        uint4 *search_zone_info,
        int num_thread_handlers = 12) : ParallelProcessor(num_thread_handlers, params[FRAME_PARAM_WIDTH], params[FRAME_PARAM_HEIGHT]),
                                        _frame(frame),
                                        _params(params),
                                        _class_costs(class_costs),
                                        _goal(goal),
                                        _wheelbase(wheelbase),
                                        _delta_max_rad(delta_max_rad),
                                        _node_conf(node_conf),
                                        _node_data(node_data),
                                        _search_zone_info(search_zone_info)
    {
        _max = params[FRAME_PARAM_WIDTH] * params[FRAME_PARAM_HEIGHT];
    }
};

class GoalWaveStep2ParallelCompute : public ParallelProcessor
{
    int _max;
    float3 *_frame;
    int *_params;
    float *_class_costs;
    int _pos;
    float3 _goal;
    float _wheelbase;
    float _delta_max_rad;
    int4 *_node_conf;
    float4 *_node_data;
    uint4 *_search_zone_info;

protected:
    void handler(int threadId) override
    {
        if (threadId >= _max)
            return;

        to_goal_wave_2(_frame, _params, _class_costs, threadId, _goal, _wheelbase, _delta_max_rad, _node_conf, _node_data, _search_zone_info);
    }

public:
    GoalWaveStep2ParallelCompute(
        float3 *frame,
        int *params,
        float *class_costs,
        float3 goal,
        float wheelbase,
        float delta_max_rad,
        int4 *node_conf,
        float4 *node_data,
        uint4 *search_zone_info,
        int num_thread_handlers = 12) : ParallelProcessor(num_thread_handlers, params[FRAME_PARAM_WIDTH], params[FRAME_PARAM_HEIGHT]),
                                        _frame(frame),
                                        _params(params),
                                        _class_costs(class_costs),
                                        _goal(goal),
                                        _wheelbase(wheelbase),
                                        _delta_max_rad(delta_max_rad),
                                        _node_conf(node_conf),
                                        _node_data(node_data),
                                        _search_zone_info(search_zone_info)
    {
        _max = params[FRAME_PARAM_WIDTH] * params[FRAME_PARAM_HEIGHT];
    }
};

void WGraph::compute_goal_wave(
    SearchFrame *frame,
    Waypoint &goal)
{
    int *_search_space_params = frame->getFrameParamsPtr();
    float *_class_costs = frame->getClassCostsPtr();
    uint4 *search_zone_info = frame->getSearchZonePtr();

    float3 goalpoint = {
        TO_FLOAT(goal.x()),
        TO_FLOAT(goal.z()),
        TO_FLOAT(goal.heading().rad())};

    GoalWaveParallelCompute compute(frame->getPtr(), _search_space_params, _class_costs, //
                                    goalpoint, _wheelbase, _max_steering_angle_rad,      //
                                    _node_conf->getPtr(), _node_data->getPtr(), search_zone_info);
    compute.runAndWait();

    GoalWaveStep2ParallelCompute compute2(frame->getPtr(), _search_space_params, _class_costs, //
                                    goalpoint, _wheelbase, _max_steering_angle_rad,      //
                                    _node_conf->getPtr(), _node_data->getPtr(), search_zone_info);
    compute2.runAndWait();

}