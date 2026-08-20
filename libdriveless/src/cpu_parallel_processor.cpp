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

extern "C"
{
    void run_parallel_func(void *(handler_fn)(int thread_id), int num_thread_handlers, int num_blocks, int num_threads_per_block)
    {
        class CppParallelProcessor : public ParallelProcessor
        {
        private:
            void *(*_handler_fn)(int thread_id);

        public:
            CppParallelProcessor(void *(handler_fn)(int thread_id), int numThreadHandlers, int numBlocks, int numThreadsPerBlock)
                : ParallelProcessor(numThreadHandlers, numBlocks, numThreadsPerBlock) , _handler_fn(handler_fn) {}

            void handler(int threadId) override
            {
                _handler_fn(threadId);
            }
        };

        CppParallelProcessor processor(handler_fn, num_thread_handlers, num_blocks, num_threads_per_block);
        processor.runAndWait();
    }
}