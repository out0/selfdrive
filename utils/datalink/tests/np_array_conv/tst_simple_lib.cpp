#include <gtest/gtest.h>
#include "simple_lib.h"

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

void printRawData(char *p, long size) {
    printf("[");
    for (long i = 0; i < size; i++)
        printf(" %d", p[i]);
    printf("]\n");
}
void printFloatData(float *p, long size) {
    printf("[");
    for (long i = 0; i < size; i++)
        printf(" %f", p[i]);
    printf("]\n");
}

TEST(FloatDataTest, GenFloatData)
{
    int s = 0;
    char * data = gen_float_data(100, &s);
    float *conv_data = new float[100];

    floatp p;
    for (int i = 0; i < 100; i++) {
        float orig = 1.0001 * i;
        int pos_data = i * sizeof(float);

        for (int k = 0; k < sizeof(float); k++) {
            p.bval[k] = data[pos_data + k];
        }

        conv_data[i] = p.val;

        ASSERT_FLOAT_EQ(orig, p.val);
    }

    printRawData(data, 100 * sizeof(float));
    printFloatData(conv_data, 100);

    free_mem(data);
    delete []conv_data;
}