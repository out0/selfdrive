
#include "../include/cuda_graph.h"

extern "C"
{
    void *load_frame(int width, int height)
    {
        return new CudaGraph(width, height);
    }

    void destroy_frame(void *self)
    {
        CudaGraph *f = (CudaGraph *)self;
        delete f;
    }

    void clear(void *self, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        f->clear();
    }

    int *find_best_neighbor(void *self, int x, int z, float radius)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->find_best_neighbor(x, z, radius);
    }

    void add_point(void *self, int x, int z, int parent_x, int parent_z, float cost)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->add_point(x, z, parent_x, parent_z, cost);
    }

    void free_waypoint(int *waypoint)
    {
        delete[] waypoint;
    }

    unsigned int count(void *self)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->count();
    }
 
    bool check_in_graph(void *self, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->checkInGraph(x, z);
    }

    int * get_parent(void *self, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->getParent(x, z);
    }

    int list_graph_points(void *self, int *points) {
        CudaGraph *f = (CudaGraph *)self;
        return f->listGraphPoints(points);
    }
}
