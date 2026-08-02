#include "../../include/search_frame.h"
#include "../../include/cpu_parallel_processor.h"
#include <stdexcept>
#include <tuple>

extern void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount);
extern __device__ __host__ bool is_zone_border(int x, int z, int xg, int zg, int search_zone_dim_w, int search_zone_dim_h);
extern __device__ __host__ int2 zone_location(int2 zone_dim_size, int2 zone_grid_size, int x, int z);

void SearchFrame::setClassColors(std::vector<std::tuple<int, int, int>> colors)
{
    if (_classCount > 0 && colors.size() != _classCount)
    {
        throw std::invalid_argument("invalid number of classed on setClassColors(). Expected: " + std::to_string(_classCount) + " obtained: " + std::to_string(colors.size()));
    }

    if (colors.size() == 0)
        return;

    if (_classCount > 0 && colors.size() != _classCount)
    {
        throw std::invalid_argument("invalid number of classed on setClassColors(). Expected: " + std::to_string(_classCount) + " obtained: " + std::to_string(colors.size()));
    }

    _classCount = colors.size();
    _classColors = std::make_unique<uchar3[]>(colors.size());

    int i = 0;
    for (auto const &c : colors)
    {
        std::tie(_classColors.get()[i].x, _classColors.get()[i].y, _classColors.get()[i].z) = c;
        i++;
    }
}

class ParallelColorExport : public ParallelProcessor
{
    float3 *_frame;
    uchar3 *_output;
    int _maxId;
    uchar3 *_classColors;
    int _classCount;
    bool _show_search_zone_marks;
    int *_search_params;
    int _width;

public:
    ParallelColorExport(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount, int numThreadHandlers, bool show_search_zone_marks, int *search_params)
        : ParallelProcessor(numThreadHandlers, width, width)
    {
        this->_frame = frame;
        this->_output = output;
        this->_maxId = width * height;
        this->_classColors = classColors;
        this->_classCount = classCount;
        this->_show_search_zone_marks = show_search_zone_marks;
        this->_search_params = search_params;
        this->_width = width;
    }

    void handler(int threadId) override
    {
        if (threadId >= _maxId)
            return;

        int pos = threadId;
        int segClass = _frame[pos].x;
        if (segClass < 0 || segClass >= _classCount)
            return;

        _output[pos].x = _classColors[segClass].x;
        _output[pos].y = _classColors[segClass].y;
        _output[pos].z = _classColors[segClass].z;

        if (_show_search_zone_marks)
        {

            const int search_zone_dim_w = _search_params[FRAME_SEARCH_ZONE_DIM_WIDTH];
            const int search_zone_dim_h = _search_params[FRAME_SEARCH_ZONE_DIM_HEIGHT];
            const int search_zone_grid_w = _search_params[FRAME_SEARCH_ZONE_GRID_WIDTH];
            const int search_zone_grid_h = _search_params[FRAME_SEARCH_ZONE_GRID_HEIGHT];

            int z = pos / _width;
            int x = pos - z * _width;

            int2 location = zone_location({search_zone_dim_w, search_zone_dim_h}, {search_zone_grid_w, search_zone_grid_h}, x, z);

            if (is_zone_border(x, z, location.x, location.y, search_zone_dim_w, search_zone_dim_h))
            {
                _output[pos].x = 128;
                _output[pos].y = 128;
                _output[pos].z = 128;
            }
        }
    }
};

bool SearchFrame::exportToColorFrame(uchar *dest, bool show_search_zone_marks)
{
    if (_classColors == nullptr)
        return false;

    int size = width() * height();
    uchar3 *ptr = new uchar3[size];

    ParallelColorExport(getPtr(), ptr, width(), height(), _classColors.get(), _classCount, _numCPUThreadHandlers, show_search_zone_marks, _params.get()).runAndWait();

    /// TODO dest podia fazer parte de ParallelColorExport
    for (int i = 0; i < size; i++)
    {
        long pos = 3 * i;
        dest[pos] = ptr[i].x;
        dest[pos + 1] = ptr[i].y;
        dest[pos + 2] = ptr[i].z;
    }

    delete[] ptr;
    return true;
}
