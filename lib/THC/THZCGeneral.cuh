#ifndef THZC_GENERALcu_INC
#define THZC_GENERALcu_INC

#include <thrust/complex.h>
#include <complex.h>

#ifndef cx
#define cx float _Complex           // used in all public interfaces
#endif

typedef thrust::complex<float> ccx; // used in kernels because of available operators
ccx toCcx(cx val);

#endif
