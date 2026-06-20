#include "../../include/search_frame_cpu.h"
#include "../../include/cpu_parallel_processor.h"
#include <stdexcept>
#include <tuple>

extern void __CUDA_KERNEL_FrameColor(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount);

void SearchFrameCPU::setClassColors(std::vector<std::tuple<int, int, int>> colors)
{   
    if (colors.size() == 0)
        return;

    if (_classCount > 0 && colors.size() != _classCount) {
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

public:
    ParallelColorExport(float3 *frame, uchar3 *output, int width, int height, uchar3 *classColors, int classCount, int numThreadHandlers)
        : ParallelProcessor(numThreadHandlers, width, width)
    {
        this->_frame = frame;
        this->_output = output; 
        this->_maxId = width * height;
        this->_classColors = classColors;
        this->_classCount = classCount;
    }

    void handler(int threadId) override
    {
        if (threadId >= _maxId)
            return;

        int pos = threadId;
        int segClass = _frame[pos].x;
        if (segClass < 0 || segClass >= _classCount) return;

        _output[pos].x = _classColors[segClass].x;
        _output[pos].y = _classColors[segClass].y;
        _output[pos].z = _classColors[segClass].z;
    }


};


bool SearchFrameCPU::exportToColorFrame(uchar *dest)
{
    if (_classColors == nullptr)
        return false;

    uchar3 *resultImgPtr = nullptr;
    if (!cudaAllocMapped(&resultImgPtr, sizeof(uchar3) * (width() * height())))
        return false;

    int size = width() * height();
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    (new ParallelColorExport(getPtr(), resultImgPtr, width(), height(), _classColors.get(), _classCount, _numCPUThreadHandlers))->runAndWait();


    for (int i = 0; i < size; i++) {
        long pos = 3 * i;
        dest[pos] = resultImgPtr[i].x;
        dest[pos+1] = resultImgPtr[i].y;
        dest[pos+2] = resultImgPtr[i].z;
    }
    
    cudaFreeHost(resultImgPtr);
    return true;
}

