#include "../src/cuda_graph.h"
#include <gtest/gtest.h>

TEST(RRTGraph, TestCreateDelete)
{
    CudaGraph *g = new CudaGraph(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3, 1);
    delete g;
}

TEST(RRTGraph, TestBasicFeatures)
{
    CudaGraph *g = new CudaGraph(100, 100, 0, 0, 0, 0, 0, 0, 10, 10, 40, 3, 1);
    
    ASSERT_EQ(0, g->count());

    g->add(50, 50, -1, -1, 0);
    g->add(50, 45, 50, 50, 10);
    g->add(55, 40, 50, 45, 10);
    g->add(20, 35, 50, 45, 10);
    g->add(20, 15, 20, 35, 10);
    g->add(20, 5, 20, 15, 10);
    g->add(20, 0, 20, 5, 10);

    ASSERT_EQ(7, g->count());

    g->remove(20, 0);
    ASSERT_EQ(6, g->count());

    // remove non existant should ignore
    g->remove(0, 0);
    ASSERT_EQ(6, g->count());

    // remove of out-of-bound node should ignore
    g->remove(999, -999);
    ASSERT_EQ(6, g->count());

    // does not exist
    int2 p = g->getParent(20, 0);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);
    
    // out-of-bound
    p = g->getParent(2000, -1000);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);
    
    // exists
    p = g->getParent(20, 5);
    ASSERT_EQ(20, p.x);
    ASSERT_EQ(15, p.y);

    // setParent of non existant node should ignore
    g->setParent(99, 99, 10, 10);
    p = g->getParent(99, 99);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);

    // setParent of non existant out-of-bound node should ignore
    g->setParent(999, 999, 10, 10);
    p = g->getParent(999, 999);
    ASSERT_EQ(-1, p.x);
    ASSERT_EQ(-1, p.y);


    // exists
    ASSERT_FLOAT_EQ(g->getCost(55, 40), 10.0);

    // doesnt exist
    ASSERT_FLOAT_EQ(g->getCost(99, 99), -1);
    
    // out-of-bound
    ASSERT_FLOAT_EQ(g->getCost(999, 999), -1);

    g->setCost(55, 40, 20.0);
    ASSERT_FLOAT_EQ(g->getCost(55, 40), 20.0);

    // doesnt exist
    g->setCost(99, 99, 100.0);
    ASSERT_FLOAT_EQ(g->getCost(99, 99), -1);

    // out-of-bound
    g->setCost(999, 999, 100.0);
    ASSERT_FLOAT_EQ(g->getCost(999, 999), -1);


    delete g;
}
