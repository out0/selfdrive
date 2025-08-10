#pragma once

#ifndef __ASYNC_COMPONENT_DRIVELESS_H
#define __ASYNC_COMPONENT_DRIVELESS_H

// CODE:BEGIN

#include <thread>
#include <memory>

class AsyncComponent {

private:
    int _period_ms;
    bool _run;
    void _auto_loop();
    std::unique_ptr<std::thread> _thr;

protected:
    AsyncComponent(int period_ms);
    ~AsyncComponent();

public:
    void start();
    void stop();
    virtual void loop(double dt) = 0;
};

// CODE:END

#endif