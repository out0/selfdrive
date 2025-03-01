#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../src/kinematic_model.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
//#include <Python.h>
#include <iostream>
#include "../include/physical_parameters.h"

// TEST(KinematicModel, TestCurveGen)
// {
//     Py_Initialize();
//     PyRun_SimpleString("print('Hello from Python!')");
//     Py_Finalize();
// }

TEST(KinematicModel, TestBuildCurves)
{
    double3 start;
    start.x = 128;
    start.y = 128;
    start.z = 0;

    double3 end;
    end.x = 230;
    end.y = 14;
    end.z = 0.0;

    double _rw = OG_WIDTH / OG_REAL_WIDTH;
    double _rh = OG_HEIGHT / OG_REAL_HEIGHT;
    double3 _center;
    _center.x = static_cast<int>(round(OG_WIDTH / 2));
    _center.y = static_cast<int>(round(OG_HEIGHT / 2));
    _center.z = 0.0;

    double _lr = 0.5 * (LOWER_BOUND_Z - UPPER_BOUND_Z) / (OG_HEIGHT / OG_REAL_HEIGHT);

    std::vector<double3> curve = CurveGenerator::buildCurveWaypoints(
        _center,
        _rw,
        _rh,
        _lr,
        MAX_STEERING_ANGLE,
        start,
        end,
        2.0);

    ASSERT_TRUE(curve.size() > 0);
}