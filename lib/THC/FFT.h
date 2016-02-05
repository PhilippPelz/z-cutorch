#ifndef FFT_H
#define FFT_H

#include "THZCTensor.h"
#include "THZCGeneral.h"

THZC_API void THZCudaTensor_fft(THCState *state, THZCudaTensor *self,
                                THZCudaTensor *result);
THZC_API void THZCudaTensor_fftBatched(THCState *state, THZCudaTensor *self,
                                       THZCudaTensor *result);

THZC_API void THZCudaTensor_ifft(THCState *state, THZCudaTensor *self,
                                 THZCudaTensor *result);
THZC_API void THZCudaTensor_ifftBatched(THCState *state, THZCudaTensor *self,
                                        THZCudaTensor *result);

THZC_API void THZCudaTensor_ifftU(THCState *state, THZCudaTensor *self,
                                  THZCudaTensor *result);
THZC_API void THZCudaTensor_ifftBatchedU(THCState *state, THZCudaTensor *self,
                                         THZCudaTensor *result);

THZC_API void THZCudaTensor_fftShiftedInplace(THCState *state,
                                              THZCudaTensor *self);
THZC_API void THZCudaTensor_fftShifted(THCState *state, THZCudaTensor *self,
                                       THZCudaTensor *result);

THZC_API void THZCudaTensor_fftShiftInplace(THCState *state,
                                            THZCudaTensor *self);
THZC_API void THZCudaTensor_fftShift(THCState *state, THZCudaTensor *self,
                                     THZCudaTensor *result);
THZC_API void THZCudaTensor_ifftShiftInplace(THCState *state,
                                             THZCudaTensor *self);
THZC_API void THZCudaTensor_ifftShift(THCState *state, THZCudaTensor *self,
                                      THZCudaTensor *result);

#endif
