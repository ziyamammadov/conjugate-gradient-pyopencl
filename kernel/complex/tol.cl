#include "cmplx.h"

__kernel void tol(__global const cfloat *dot, __global const float *tolerance,
                  __global int *flags, const int nRhs, const int workGroups) {

  const int r = get_global_id(0);
  if (r < nRhs) {
    cfloat sum = (cfloat)(0.0f, 0.0f);
    for (int i = 0; i < workGroups; ++i)
      sum = cadd(sum, dot[r * workGroups + i]);

    flags[r] = cnorm2(sum) < tolerance[0] ? 1 : 0;
  }
}
