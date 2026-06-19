#pragma once
#include <vector>
#include <thread>

class ParallelProcessor {
private:
    int _numThreadPerHandler;
    int _numThreadHandlers;
    bool _running;
    std::vector<std::thread> _threads;
    void _threadRun(int handlerId);
    void _clearAll();

public:
    ParallelProcessor(
        int numThreadHandlers,
        int numBlocks,
        int numThreadsPerBlock
    );
    ~ParallelProcessor();

    

    virtual void handler(int threadId) = 0;

    void runAndWait();
};