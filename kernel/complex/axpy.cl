
#include "cmplx.h"

__kernel void axpy(__global cfloat *x, __global cfloat *y, __constant cfloat *a,
                   const int aSign, const int size, const int nRhs) {

  const int __idx = get_global_id(0);
  if (__idx < size) {
    if (aSign)
      for (int r = 0; r < nRhs; r++)
        y[__idx + r * size] =
            cadd(y[__idx + r * size], cmul(a[r], x[__idx + r * size]));
    else
      for (int r = 0; r < nRhs; r++)
        y[__idx + r * size] =
            csub(y[__idx + r * size], cmul(a[r], x[__idx + r * size]));
  }
}