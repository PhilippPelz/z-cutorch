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

#define cufftSafeCall(err)      __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line) {
	if (CUFFT_SUCCESS != err) {
		printf("CUFFT error in file '%s', line %d\n %s\nerror: %d\nterminating!\n",
		file, line, err, _cudaGetErrorEnum(err));
		cudaDeviceReset();
		// assert(0);
	}
}
// THC_API int THCState_getNumCuFFTPlans(THCState* state);
// THC_API cufftHandle* THCState_getCuFFTPlan(THCState* state,int batch, int n1, int n2, int n3) ;
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
	int fft_dims[3] = {0,0,0};
	for (int i = 1; i < ndim; i++) {
		fft_dims[i - 1] = (int) THZCudaTensor_size(state, self, i);
	}
	// printf("here 1\n");
	cufftHandle handle;
	// printf("%d %d %d \n",fft_dims[0],fft_dims[1],fft_dims[2]);
	// THCState_getCuFFTPlan(state,&plan,batch,1,2,3);
	int n1 = fft_dims[0];
	int n2 = fft_dims[1];
	int n3 = fft_dims[2];

	int planExists = 0;
	for(int i = 0; i<state->numCuFFTPlans; i++){
		THCuFFTPlan plan = state->cuFFTPlans[i];
		if(plan.n1 == n1 && plan.n2 == n2 && plan.n3 == n3 && batch == plan.batches){
			// printf("plan exists \n");
			handle = plan.plan;
			cufftSafeCall(cufftSetStream(plan.plan, THCState_getCurrentStream(state)));
			planExists = 1;
		}
	}
	// printf("wp2\n");
	if(planExists == 0){
		int fft_dims[3] = {n1,n2,n3};
		int ndim = 0;
		if(n1>0) ndim++;
		if(n2>0) ndim++;
		if(n3>0) ndim++;

		if (state->numCuFFTPlans + 1 > state->maxCuFFTPlans)
			printf("Make state->maxCuFFTPlans larger, memory leaks are created now.");
		state->numCuFFTPlans = (state->numCuFFTPlans + 1) > state->maxCuFFTPlans ? state->numCuFFTPlans : state->numCuFFTPlans + 1;

		cufftSafeCall(cufftPlanMany(&(state->cuFFTPlans[state->numCuFFTPlans].plan), ndim, fft_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch));
		// printf("here 2\n");
		// printf("wp3\n");
		cufftSafeCall(cufftSetStream(state->cuFFTPlans[state->numCuFFTPlans].plan, THCState_getCurrentStream(state)));

		state->cuFFTPlans[state->numCuFFTPlans].n1 = n1;
		state->cuFFTPlans[state->numCuFFTPlans].n2 = n2;
		state->cuFFTPlans[state->numCuFFTPlans].n3 = n3;
		state->cuFFTPlans[state->numCuFFTPlans].batches = batch;

		handle = state->cuFFTPlans[state->numCuFFTPlans].plan;
	}

	// cufftSafeCall(cufftPlanMany(&plan, ndim - 1, fft_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, batch));
	// printf("here 2\n");
	// cufftSafeCall(cufftSetStream(plan, THCState_getCurrentStream(state)));
	// printf("here 3\n");
	cufftSafeCall(cufftExecC2C(handle, (cufftComplex *)THZCudaTensor_data(state, self), (cufftComplex *)THZCudaTensor_data(state, result), direction));
	// printf("here 4\n");
	// cufftDestroy(plan);
	// printf("here 5\n");
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
