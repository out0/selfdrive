#include <gtest/gtest.h>
#include "../../include/quaternion.h"
#include "test_utils.h"
#include <cmath>
#include <iostream>

TEST(QuaternionTst, TestBasicProperties)
{
    quaternion x(0, 1, 0, 0);
    quaternion y(0, 0, 1, 0);
    quaternion z(0, 0, 0, 1);

    ASSERT_EQ(x * y, z);
    ASSERT_EQ(y * z, x);    
    ASSERT_EQ(z * x, y);

    ASSERT_EQ(y * x, -z);
    ASSERT_EQ(z * y, -x);    
    ASSERT_EQ(x * z, -y);

    ASSERT_EQ(x * 2, quaternion(0, 2, 0, 0));
    ASSERT_EQ(x + 2, quaternion(2, 1, 0, 0));


    ASSERT_EQ(quaternion(), quaternion(1, 0, 0, 0));
}

    // inline void set(double w, double x, double y, double z) {
    //     *data = {w, x, y, z};
    // }



TEST(QuaternionTst, TestYawRotation)
{
    quaternion q1 (0, 1, 0, 0);

    // rotating the x axis around Z
    for (int i = 1; i <= 359; i++) {
        q1.rotate_z(angle::deg(1));
        ASSERT_EQ(angle::deg(i), q1.yaw());
    }

    quaternion q2 (0, 1, 0, 0);
    // rotating the x axis around Z backwards
    for (int i = 359; i >= 0; i--) {
        q2.rotate_z(angle::deg(-1));
        ASSERT_EQ(angle::deg(i), q2.yaw());
    }

    quaternion q3 (0, 1, 0, 0);
    q3.rotate_z(angle::deg(90));
    ASSERT_EQ(quaternion(0, 0, 1, 0), q3);

    quaternion q4 (0, 1, 0, 0);
    q4.rotate_z(angle::deg(-90));
    //std::cout << q4.to_string() << "\n";
    ASSERT_EQ(quaternion(0, 0, -1, 0), q4);
}


TEST(QuaternionTst, TestPitchRotation)
{
    quaternion q1 (0, 1, 0, 0);
    q1.rotate_pitch(angle::deg(90));
    ASSERT_EQ(quaternion(0, 0, 0, -1), q1);
    auto a1 = q1.pitch();
//    std::cout << "q1 = " << q1 << ", a = " << a1 << "\n";
    ASSERT_EQ(a1, angle::deg(90));

    quaternion q2 (0, 1, 0, 0);
    q2.rotate_pitch(angle::deg(-90));
  //  std::cout << q2 << "\n";
    ASSERT_EQ(quaternion(0, 0, 0, 1), q2);

    quaternion q3 (0, 1, 0, 0);
    //rotating the x axis around Y
    for (int i = 1; i <= 359; i++) {
        q3.rotate_y(angle::deg(1));

        ASSERT_EQ(angle::deg(i), q3.pitch());
    }

    quaternion q4 (0, 1, 0, 0);
    // rotating the x axis around Z backwards
    for (int i = 359; i >= 0; i--) {
        q4.rotate_y(angle::deg(-1));
        ASSERT_EQ(angle::deg(i), q4.pitch());
    }
}


TEST(QuaternionTst, TestRollRotation)
{
    quaternion q1 (0, 0, 1, 0);
    q1.rotate_roll(angle::deg(90));
    ASSERT_EQ(quaternion(0, 0, 0, 1), q1);
    auto a1 = q1.roll();
    //std::cout << "q1 = " << q1 << ", a = " << a1 << "\n";
    ASSERT_EQ(a1, angle::deg(90));

    quaternion q2 (0, 0, 1, 0);
    q2.rotate_roll(angle::deg(-90));
    ASSERT_EQ(quaternion(0, 0, 0, -1), q2);
    auto a2 = q2.roll();
    //std::cout << "q2 = " << q2 << ", a = " << a2 << "\n";
    ASSERT_EQ(a2, angle::deg(270));

    quaternion q3 (0, 0, 1, 0);
    //rotating the x axis around Y
    for (int i = 1; i <= 359; i++) {
        q3.rotate_x(angle::deg(1));

        ASSERT_EQ(angle::deg(i), q3.roll());
    }

    quaternion q4 (0, 0, 1, 0);
    // rotating the x axis around Z backwards
    for (int i = 359; i >= 0; i--) {
        q4.rotate_x(angle::deg(-1));
        ASSERT_EQ(angle::deg(i), q4.roll());
    }
}


TEST(QuaternionTst, TestEulerAngleInit)
{
    quaternion q1(angle::deg(0), angle::deg(0), angle::deg(0));
    ASSERT_EQ(q1, quaternion(0, 1, 0, 0));

    quaternion q2(angle::deg(45), angle::deg(0), angle::deg(0));
    ASSERT_EQ(q2, quaternion(0, sqrt(2)/2, sqrt(2)/2, 0));

    quaternion q3(angle::deg(90), angle::deg(0), angle::deg(90));
    ASSERT_EQ(q3, quaternion(0, 0, 0, 1));

}

__global__ static void __test_kernel_quaternion(double4 *frame, int width, int height)
{
    int pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos >= width * height)
        return;

    if (pos == 0) {
        printf("rotating start %f, %f, %f, %f\n", frame[pos].w, frame[pos].x, frame[pos].y, frame[pos].z);
    }
    quaternion_rotate_z(&frame[pos], &frame[pos], PI/2);
    if (pos == 0) {
        printf("rotating result %f, %f, %f, %f\n", frame[pos].w, frame[pos].x, frame[pos].y, frame[pos].z);
    }

}


TEST(QuaternionTst, TestCudaExec)
{
    return;
    int width = 100;
    int height = 100;
    int size = width * height;
    int numBlocks = floor(size / THREADS_IN_BLOCK) + 1;

    double4 *frame;
    cudaAllocMapped(&frame, sizeof(double4) * width * height);

    for (int i = 0; i < size; i++) {
        quaternion q(&frame[i]);
        q.set(0, 1, 0, 0);
    }

    __test_kernel_quaternion<<<numBlocks, THREADS_IN_BLOCK>>>(frame, width, height);
    CUDA(cudaDeviceSynchronize());

    for (int i = 0; i < size; i++) {
        quaternion q(&frame[i]);
        ASSERT_EQ(q, quaternion(0, 0, 1, 0));
    }


    cudaFree(frame);

}