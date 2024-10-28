
#include "../include/cuda_graph.h"
#include "../include/cuda_frame.h"

extern "C"
{
    void *load_frame(
        int width, 
        int height,
        int min_dist_x,
        int min_dist_z,
        int lower_bound_ego_x,
        int lower_bound_ego_z,
        int upper_bound_ego_x,
        int upper_bound_ego_z)
    {
        return new CudaGraph(
            width, 
            height,
            min_dist_x,
            min_dist_z,
            lower_bound_ego_x,
            lower_bound_ego_z,
            upper_bound_ego_x,
            upper_bound_ego_z);
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

    int *find_nearest_neighbor(void *self, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->find_nearest_neighbor(x, z);
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

    int *get_parent(void *self, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        return f->getParent(x, z);
    }

    void link(void *self, void* cuda_frame, int parent_x, int parent_z, int x, int z)
    {
        CudaGraph *f = (CudaGraph *)self;
        CudaFrame *cudaf = (CudaFrame *)cuda_frame;
        return f->link(cudaf->getFramePtr(), parent_x, parent_z, x, z);
    }

    void list_nodes(void *self, float *res, int count) {
        CudaGraph *f = (CudaGraph *)self;
        f->listNodes(res, count);
    }


    // int list_graph_points(void *self, int *points) {
    //     CudaGraph *f = (CudaGraph *)self;
    //     return f->listGraphPoints(points);
    // }
}
