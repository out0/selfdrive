#include "wpmp_data.h"

__device__ __host__ void SET_NODE_PARENT(int4 *node_conf, int pos, int parent_x, int parent_z)
{
    node_conf[pos].x = parent_x;
    node_conf[pos].y = parent_z;
}