#pragma once

#ifndef __GRAPH_WPMP
#define __GRAPH_WPMP

#include <driveless/cuda_basic.h>
#include <driveless/frame_params.h>

typedef struct SE_conf
{
    float3 *frame;
    int *params;
    float *class_costs;
    float wheelbase;
    float delta_max_rad;
    int2 min_distances;
} SE_conf;

typedef struct graph_conf
{
    int4 *node_conf;
    float4 *node_data;
} graph_conf;

#define NODE_HEADING(node_data, pos) (node_data[pos].x)
#define SET_NODE_HEADING(node_data, pos, val) (node_data[pos].x = val)

#define NODE_COST_TO_GOAL(node_data, pos) (node_data[pos].y)
#define SET_NODE_COST_TO_GOAL(node_data, pos, val) (node_data[pos].y = val)

#define NODE_COST_FROM_START(node_data, pos) (node_data[pos].z)
#define SET_NODE_COST_FROM_START(node_data, pos, val) (node_data[pos].z = val)

#define NODE_TYPE(node_conf, pos) (node_conf[pos].z)
#define SET_NODE_TYPE(node_conf, pos, val) (node_conf[pos].z = val)

#define NODE_PARENT(node_conf, pos) ({node_conf[pos].x, node_conf[pos].y})
extern __device__ __host__ void SET_NODE_PARENT(int4 *node_conf, int pos, int parent_x, int parent_z);

#define COMPUTE_POS(width, x, z) (z * width + x)

#define NODE_TYPE_NULL 0
#define NODE_TYPE_GRAPH 1
#define NODE_TYPE_ORIGIN 2
#define NODE_TYPE_GRAPH_CONNECTED_TO_GOAL 3
#define NODE_TYPE_NULL_CONNECTED_TO_GOAL 4

#define IS_NODE(node_conf, pos) (node_conf[pos].z == NODE_TYPE_GRAPH || node_conf[pos].z == NODE_TYPE_ORIGIN || node_conf[pos].z == NODE_TYPE_GRAPH_CONNECTED_TO_GOAL)

#endif