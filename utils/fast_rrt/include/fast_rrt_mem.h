#ifndef H_FAST_RRT_MEM
#define H_FAST_RRT_MEM
#include <stdio.h>

template <typename T>
class Memlist {
public:    
    T *data;
    unsigned int size;

    Memlist() {
        data = nullptr;
    }

    ~Memlist() {
        if (data != nullptr) {
            printf ("deleting data\n");
            delete []data;
        }
    }
};

#endif