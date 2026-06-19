#include "../include/cpu_parallel_processor.h"
#include <thread>
#include <vector>
#include <cmath>

ParallelProcessor::ParallelProcessor(
    int numThreadHandlers,
    int numBlocks,
    int numThreadsPerBlock)
{
    int _numVirtualThreads = numBlocks * numThreadsPerBlock;
    _numThreadPerHandler = int(ceil(_numVirtualThreads / (double)numThreadHandlers));
    _numThreadHandlers = numThreadHandlers;
}
ParallelProcessor::~ParallelProcessor()
{
    _running = false;
    _clearAll();
}

void ParallelProcessor::_clearAll()
{
    for (auto &thread : _threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    _threads.clear();
}

void ParallelProcessor::_threadRun(int handlerId)
{

    int queue = 0;
    while (_running && queue < _numThreadPerHandler)
    {
        int threadId = handlerId * _numThreadPerHandler + queue;
        handler(threadId);
        queue++;
    }
}

void ParallelProcessor::runAndWait()
{
    _running = true;
    for (int i = 0; i < _numThreadHandlers; ++i)
    {
        _threads.emplace_back(&ParallelProcessor::_threadRun, this, i);
    }
    _clearAll();
    _running = false;
}
