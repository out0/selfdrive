#include "../../include/search_frame.h"
#include <driveless/cpu_parallel_processor.h>

extern __device__ __host__ void count_obstacle_in_search_zones(float3 *frame, float *classCosts, int *search_params, int2 search_zone_dim, uint2 *search_zone_info, int pos);

void SearchFrame::initialize_search_zones()
{
    int W = TO_INT(round(static_cast<double>(width()) / static_cast<double>(_searchZoneDim.first))) + 1;
    int H = TO_INT(round(static_cast<double>(height()) / static_cast<double>(_searchZoneDim.second))) + 1;
    _search_zone_info = std::make_unique<Frame<uint2>>(W, H, _numCPUThreadHandlers);
}

class PreComputeSearchZonesProcess : public ParallelProcessor
{
    int _max;
    float3 *_frame;
    float *_classCosts;
    int *_search_params;
    int2 _search_zone_dim;
    uint2 *_search_zone_info;
    int _num_thread_handlers;

protected:
    void handler(int threadId) override
    {
        if (threadId >= _max)
            return;

        count_obstacle_in_search_zones(_frame, _classCosts, _search_params, _search_zone_dim, _search_zone_info, _num_thread_handlers);
    }

public:
    PreComputeSearchZonesProcess(float3 *frame,
                                 float *classCosts,
                                 int *search_params,
                                 int2 search_zone_dim,
                                 uint2 *search_zone_info,
                                 int num_thread_handlers) : ParallelProcessor(num_thread_handlers, search_params[FRAME_PARAM_WIDTH], search_params[FRAME_PARAM_HEIGHT]),
                                                            _frame(frame),
                                                            _classCosts(classCosts),
                                                            _search_params(search_params),
                                                            _search_zone_dim(search_zone_dim),
                                                            _search_zone_info(search_zone_info),
                                                            _num_thread_handlers(num_thread_handlers)
    {
        _max = search_params[FRAME_PARAM_WIDTH] * search_params[FRAME_PARAM_HEIGHT];
    }
};

void SearchFrame::pre_compute_search_zones()
{
    int2 _search_zone_dim = { _searchZoneDim.first, _searchZoneDim.second };

    PreComputeSearchZonesProcess(getPtr(), _classCosts.get(), _params.get(), _search_zone_dim, _search_zone_info->getPtr(), _numCPUThreadHandlers)
        .runAndWait();
}