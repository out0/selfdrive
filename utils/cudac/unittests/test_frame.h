#include <gtest/gtest.h>
#include "../cuda_frame.h"
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

class TestFrame
{
    float *data;
    int width;
    int height;

    int getVectorPos(int x, int z);
    void addPoint(int x, int z, int r, int g, int b);


public:
    TestFrame(int width, int height);
    ~TestFrame();

    void addObstacle(int x1, int z1, int x2, int z2);
    void addBluePoint(int x, int z);
    void addRedPoint(int x, int z);
    void addGreenPoint(int x, int z);
    void addVector(int x, int z, float heading, int size);
    float getDistanceCost(int x, int z);
    int getAllowedHeadings(int x, int z);

    CudaFrame *getFrame(int min_dist_x, int min_dist_z, int lower_bound_x, int lower_bound_z, int upper_bound_x, int upper_bound_z);
    float * getImgPtr();

    void toFile(const std::string &filename);
};