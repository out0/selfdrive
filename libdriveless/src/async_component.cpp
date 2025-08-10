#include "../include/async_component.h"
#include <chrono>
using std::chrono::high_resolution_clock;
using namespace std::chrono_literals;
using std::chrono::duration_cast;
using std::chrono::duration;

void AsyncComponent::_auto_loop()
{
    double last_dt = 0;
    while(_run) {
        std::this_thread::sleep_for(std::chrono::milliseconds(_period_ms));
        auto t1 = high_resolution_clock::now();
        loop(last_dt);
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> dt = t2 - t1;
        last_dt = dt.count();
    }
}

AsyncComponent::AsyncComponent(int period_ms)
{
    this->_period_ms = period_ms;
    this->_run = false;
    this->_thr = nullptr;
}
AsyncComponent::~AsyncComponent()
{
    stop();
}

void AsyncComponent::start()
{
    _run = true;
    _thr = std::make_unique<std::thread>(&AsyncComponent::_auto_loop, this);
}
void AsyncComponent::stop()
{
    if (this->_thr == nullptr)
        return;

    _run = false;
    if (this->_thr->joinable())
        this->_thr->join();

    this->_thr = nullptr;
}
