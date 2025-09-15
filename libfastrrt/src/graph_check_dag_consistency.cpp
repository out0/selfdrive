
#include <driveless/cuda_basic.h>
#include <driveless/cuda_params.h>
#include "../include/graph.h"

inline bool sameNode(int2 a, int2 b)
{
    return a.x == b.x && a.y == b.y;
}

void CudaGraph::__printInconsistentChain(int2 n, int maxLoop)
{
    int2 p;
    printf("[GRAPH check] Inconsistent DAG from %d, %d: (%d, %d)", n.x, n.y, n.x, n.y);
    p.x = n.x;
    p.y = n.y;
    for (int i = 0; i < maxLoop; i++)
    {
        int2 parent = getParent(p.x, p.y);
        if (parent.x == -1)
            return;
        printf("->(%d, %d)", parent.x, parent.y);
        p.x = parent.x;
        p.y = parent.y;
    }
    printf("\n");
}

bool CudaGraph::checkGraphIsDAG()
{
    int maxLoop = count(GRAPH_TYPE_NODE) + 2;
    std::vector<int2> nodes = list();

    int2 p;

    for (auto n : nodes)
    {
        p.x = n.x;
        p.y = n.y;
        int i = maxLoop;
        while (i-- > 0)
        {
            int2 parent = getParent(p.x, p.y);
            if (sameNode(p, parent))
            {
                __printInconsistentChain(n, maxLoop);
                return false;
            }
            if (parent.x == -1)
                break;
            p.x = parent.x;
            p.y = parent.y;
        }
        if (i <= 0)
        {
            __printInconsistentChain(n, maxLoop);
            return false;
        }
    }
    return true;
}