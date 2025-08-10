#include <gtest/gtest.h>
#include "../../include/async_component.h"
#include "test_utils.h"
#include <cmath>
#include <chrono>

class Comp : public AsyncComponent
{
public:
    int upd;
    Comp() : AsyncComponent(10)
    {
        upd = 0;
    }
    void loop(double dt) override
    {
        upd++;
    }
};

TEST(AsyncComponent, TestThreadRun)
{
    int num_thr = 100;
    Comp list[num_thr];

    for (int i = 0; i < num_thr; i++)
    {
        if (list[i].upd != 0)
            FAIL();
    }

    for (int i = 0; i < num_thr; i++)
    {
        list[i].start();
    }

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(300ms);

    for (int i = 0; i < num_thr; i++)
    {
        list[i].stop();
    }

    for (int i = 0; i < num_thr; i++)
    {
        if (list[i].upd == 0)
            FAIL();
    }
}