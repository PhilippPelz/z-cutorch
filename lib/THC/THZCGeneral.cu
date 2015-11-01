#include "THZCGeneral.cuh"

ccx toCcx(cx val) {
	return ccx(crealf(val), cimagf(val));
}
