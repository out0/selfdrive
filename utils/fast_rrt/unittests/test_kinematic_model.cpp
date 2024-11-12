#include "../src/kinematic_model.h"
#include <gtest/gtest.h>

TEST(KinematicModel, TestMemlist)
{
    Memlist<float> *list = new Memlist<float>();
    delete list;

    list = new Memlist<float>();
    list->data = new float[10000];
    list->size = 10000;
    delete list;
    
}
