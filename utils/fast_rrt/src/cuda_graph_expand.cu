#include "cuda_basic.h"
#include "class_def.h"
#include <math_constants.h>

__global__ void setup_kernel(curandState *state, long long seed){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}