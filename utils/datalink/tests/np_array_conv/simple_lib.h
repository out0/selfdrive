#ifndef __SIMPLE_LIB_H
#define __SIMPLE_LIB_H

typedef union floatp {
    float val;
    char bval[sizeof(float)];
} floatp;

char * gen_float_data(long size, int *data_word_size);
void free_mem(char *p);
void simulate_send_float_array(float *data, size_t size);

#endif