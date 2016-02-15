#include "THZCTensorMath.h"
#include "FFT.h"
#include <cufft.h>
#include <cufftXt.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "THZCGeneral.h"
#include "THZCGeneral.cuh"

// #include "arrayfire.h"

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
		case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

		case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

		case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

		case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

		case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

		case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

		case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

		case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

		case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

		case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown>";
}
#endif

void cufftShift_2D_kernel(ccx* d, int nx, int ny)
{
	// int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // // 2D Slice & 1D Line
    // int sLine = N;
    // int sSlice = N * N;
		//
    // // Transformations Equations
    // int sEq1 = (sSlice + sLine) / 2;
    // int sEq2 = (sSlice - sLine) / 2;
		//
    // // Thread Index (1D)
    // int xThreadIdx = threadIdx.x;
    // int yThreadIdx = threadIdx.y;
		//
    // // Block Width & Height
    // int blockWidth = blockDim.x;
    // int blockHeight = blockDim.y;
		//
    // // Thread Index (2D)
    // int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    // int yIndex = blockIdx.y * blockHeight + yThreadIdx;
		//
    // // Thread Index Converted into 1D Index
    // int index = (yIndex * N) + xIndex;
		//
    // T regTemp;
		//
    // if (xIndex < N / 2)
    // {
    //     if (yIndex < N / 2)
    //     {
    //         regTemp = data[index];
		//
    //         // First Quad
    //         data[index] = data[index + sEq1];
		//
    //         // Third Quad
    //         data[index + sEq1] = regTemp;
    //     }
    // }
    // else
    // {
    //     if (yIndex < N / 2)
    //     {
    //         regTemp = data[index];
		//
    //         // Second Quad
    //         data[index] = data[index + sEq2];
		//
    //         // Fourth Quad
    //         data[index + sEq2] = regTemp;
    //     }
    // }
}

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	if (CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\n %s\nerror: %d\nterminating!\n",
		__FILE__, __LINE__, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
		// assert(0);
	}
}
void THZCudaTensor_fftbase(THCState *state, THZCudaTensor *self, THZCudaTensor *result, int direction) {
	int ndim = THZCudaTensor_nDimension(state, result);
	// printf("ndim %d\n",ndim);
	// printf("pself %p\n",(void*)THZCudaTensor_data(state, self));
	// printf("presult %p\n",(void*)THZCudaTensor_data(state, result));
	int batch = 1;
	int *fft_dims = (int*)malloc(ndim*sizeof(int));
	for (int i = 0; i < ndim; i++) {
		fft_dims[i] = (int) THZCudaTensor_size(state, self, i);
		// printf("dim %d: %d\n",i,fft_dims[i]);
	}
	cufftHandle plan;
	// printf("here 1\n");
	cufftSafeCall(cufftPlanMany(&plan, ndim, fft_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch));
	// printf("here 2\n");
	cufftSafeCall(cufftSetStream(plan, THCState_getCurrentStream(state)));
	// printf("here 3\n");
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)THZCudaTensor_data(state, self), (cufftComplex *)THZCudaTensor_data(state, result), direction));
	// printf("here 4\n");
	cufftDestroy(plan);
	free(fft_dims);
}
void THZCudaTensor_fftBatchedbase(THCState *state, THZCudaTensor *self, THZCudaTensor *result, int direction) {
	int ndim = THZCudaTensor_nDimension(state, self);
	int batch = THZCudaTensor_size(state, self, 0);
	int fft_dims[3];
	for (int i = 1; i < ndim; i++) {
		fft_dims[i - 1] = (int) THZCudaTensor_size(state, self, i);
	}
	cufftHandle plan;
	cufftSafeCall(cufftPlanMany(&plan, ndim - 1, fft_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch));
	cufftSafeCall(cufftSetStream(plan, THCState_getCurrentStream(state)));
	cufftSafeCall(cufftExecC2C(plan, (cufftComplex *)THZCudaTensor_data(state, self), (cufftComplex *)THZCudaTensor_data(state, result), direction));
	cufftDestroy(plan);
}

void THZCudaTensor_fft(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_fftbase(state, self, result, CUFFT_FORWARD);
}

void THZCudaTensor_fftBatched(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_fftBatchedbase(state, self, result, CUFFT_FORWARD);
}

void THZCudaTensor_ifft(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_ifftU(state, self, result);
	THZCudaTensor_mul(state, result, result, (1 / THZCudaTensor_nElement(state, result)) + 0 * _Complex_I);
}

void THZCudaTensor_ifftBatched(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_ifftBatchedU(state, self, result);
	THZCudaTensor_mul(state, result, result, (1 / THZCudaTensor_nElement(state, result)) + 0* _Complex_I);
}


void THZCudaTensor_ifftU(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_fftbase(state, self, result, CUFFT_INVERSE);
}

void THZCudaTensor_ifftBatchedU(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	THZCudaTensor_fftBatchedbase(state, self, result, CUFFT_INVERSE);
}

void THZCudaTensor_fftShiftInplace(THCState *state, THZCudaTensor *self) {

}
void THZCudaTensor_fftShift(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	// int ndim = THZCudaTensor_nDimension(s, self);
	// int dims[4];
	// for (int i = 0; i < ndim; i++) {
	// 	dims[i] = (int) THZCudaTensor_size(s, self, i);
	// }
	// dim4 dims((const unsigned) ndim, (const long long *) dims);
	// array a(dims, (af::cfloat*)THZCudaTensor_data(state, self),af::afDevice);
	// array out = af::shift(a, a.dims(0)/2, a.dims(1)/2);

}
void THZCudaTensor_ifftShiftInplace(THCState *state, THZCudaTensor *self) {

}
void THZCudaTensor_ifftShift(THCState *state, THZCudaTensor *self, THZCudaTensor *result) {
	// int ndim = THZCudaTensor_nDimension(s, self);
	// int dims[4];
	// for (int i = 0; i < ndim; i++) {
	// 	dims[i] = (int) THZCudaTensor_size(s, self, i);
	// }
	// dim4 dims((const unsigned) ndim, (const long long *) dims);
	// array a(dims, (af::cfloat*)THZCudaTensor_data(state, self),af::afDevice);
	// array out = af::shift(wave, (a.dims(0)+1)/2, (a.dims(1)+1)/2);
}
