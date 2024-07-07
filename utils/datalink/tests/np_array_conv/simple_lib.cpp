#include <stdio.h>
#include "simple_lib.h"

extern "C"
{
    char *py_gen_float_data(long size, int *word_size)
    {
        return gen_float_data(size, word_size);
    }
    void py_free_mem(char *p)
    {
        free_mem(p);
    }

    void py_simulate_send_float_array(float *data, size_t size)
    {
        simulate_send_float_array(data, size);
    }
}

char *gen_float_data(long size, int *data_word_size)
{
    *data_word_size = sizeof(float);
    char *res = new char[size * sizeof(float)];
    floatp r;
    for (long i = 0; i < size; i++)
    {
        r.val = 1.0001 * i;

        for (int k = 0; k < sizeof(float); k++)
        {
            res[sizeof(float) * i + k] = r.bval[k];
        }
    }
    return res;
}

void simulate_send_float_array(float *data, size_t size)
{
    printf("[");
    for (long i = 0; i < size; i++)
        printf(" %f", data[i]);
    printf("]\n");
}

void free_mem(char *p)
{
    delete[] p;
}

char *gen_uint8_data(long size)
{
    return nullptr;
}
