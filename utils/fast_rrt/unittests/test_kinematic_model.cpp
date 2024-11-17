#include "../src/cuda_graph.h"
#include "../src/cuda_frame.h"
#include "../src/kinematic_model.h"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <chrono>
#include <Python.h>
#include <iostream>

TEST(KinematicModel, TestCurveGen)
{
    Py_Initialize();
    PyRun_SimpleString("print('Hello from Python!')");
    Py_Finalize();
}