CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake ${CMAKE_MODULE_PATH})

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch)
ENDIF()
IF(NOT CUDA_FOUND)
  FIND_PACKAGE(CUDA 5.5 REQUIRED)
ENDIF()
IF(NOT MAGMA_FOUND)
  FIND_PACKAGE(MAGMA)
ENDIF()

IF(MAGMA_FOUND)
  SET(USE_MAGMA 1)
  MESSAGE(STATUS "Compiling with MAGMA support")
ENDIF(MAGMA_FOUND)

IF(APPLE)
  IF(${CUDA_VERSION} LESS 6.0)
    # work around for mac os x bug:
    # http://stackoverflow.com/questions/16286588/cuda-5-0-cmake-and-make-failing-on-osx-10-8-3
    if (NOT DEFINED CUDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(CUDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by NVCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
    endif()

    # bug on Apple
    LINK_DIRECTORIES("/usr/local/cuda/lib/")
  ELSEIF(${CUDA_VERSION} LESS 7.0)
    SET(CUDA_HOST_COMPILER clang)
    LIST(APPEND CUDA_NVCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
    IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    ENDIF()
  ENDIF()
  # CUDA 7 supports clang and libc++ so no need to change anything
ENDIF(APPLE)

# Detect CUDA architecture and get best NVCC flags
INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/FindCudaArch.cmake)
SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
LIST(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_EXTRA})

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${CUDA_SDK_ROOT_DIR}/common/inc")
IF(USE_MAGMA)
  INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
ENDIF(USE_MAGMA)

IF(NOT THZC_INSTALL_BIN_SUBDIR
    OR NOT THZC_INSTALL_LIB_SUBDIR
    OR NOT THZC_INSTALL_INCLUDE_SUBDIR
    OR NOT THZC_INSTALL_CMAKE_SUBDIR)

  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/TH)
  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/THC)
  LINK_DIRECTORIES(${TH_LIB_PATH})

  SET(THZC_INSTALL_BIN_SUBDIR "bin" CACHE PATH "THZC install binary subdirectory")
  SET(THZC_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THZC install library subdirectory")
  SET(THZC_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "THZC install include subdirectory")
  SET(THZC_INSTALL_CMAKE_SUBDIR "share/cmake/THZC" CACHE PATH "THZC install cmake subdirectory")
ELSE()
  SET(THZC_INSTALL_BIN_SUBDIR ${Torch_INSTALL_BIN_SUBDIR})
  SET(THZC_INSTALL_LIB_SUBDIR ${Torch_INSTALL_LIB_SUBDIR})
  SET(THZC_INSTALL_INCLUDE_SUBDIR ${Torch_INSTALL_INCLUDE_SUBDIR})
  SET(THZC_INSTALL_CMAKE_SUBDIR ${Torch_INSTALL_CMAKE_SUBDIR})
ENDIF()
INCLUDE_DIRECTORIES("/home/philipp/torch/install/include/THC")
INCLUDE_DIRECTORIES("${THZC_INSTALL_INCLUDE_SUBDIR}/THC")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THZCGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THZCGeneral.h")

SET(CMAKE_C_FLAGS "-std=c99")
SET(src
    THZCGeneral.c THZCAllocator.c THZCStorage.c THZCStorageCopy.c THZCTensor.c THZCTensorCopy.c)

SET(src-cuda
  THZCReduceApplyUtils.cu
  THZCBlas.cu
  THZCGeneral.cu
  FFT.cu
  THZCStorage.cu
  THZCStorageCopy.cu
  THZCTensor.cu
  THZCTensorCopy.cu
  THZCTensorMath.cu
  THZCTensorMath2.cu
  THZCTensorMathBlas.cu
  THZCTensorMathMagma.cu
  THZCTensorMathPairwise.cu
  THZCTensorMathPointwise.cu
  THZCTensorMathScan.cu
  THZCTensorMathTransformReduce.cu
  THZCTensorMasked.cu
  THZCTensorIndex.cu
  THZCTensorConv.cu
  THZCTensorScatterGather.cu
  THZCApply.cu
  )

CUDA_ADD_LIBRARY(THZC SHARED ${src} ${src-cuda})
CUDA_ADD_CUBLAS_TO_TARGET(THZC)
CUDA_ADD_CUFFT_TO_TARGET(THZC)
TARGET_LINK_LIBRARIES(THZC TH ${CUDA_curand_LIBRARY})

IF(USE_MAGMA)
  TARGET_LINK_LIBRARIES(THZC ${MAGMA_LIBRARIES})
ENDIF(USE_MAGMA)

INSTALL(TARGETS THZC
          RUNTIME DESTINATION "${THZC_INSTALL_BIN_SUBDIR}"
          LIBRARY DESTINATION "${THZC_INSTALL_LIB_SUBDIR}"
          ARCHIVE DESTINATION "${THZC_INSTALL_LIB_SUBDIR}")

INSTALL(FILES
          THZC.h
          ${CMAKE_CURRENT_BINARY_DIR}/THZCGeneral.h
          FFT.h
          THZCGeneral.cuh
          THZCBlas.h
          THZCStorage.h
          THZCStorageCopy.h
          THZCTensor.h
          THZCTensorCopy.h
          THZCTensorMath.h
          THZCTensorConv.h
          THZCApply.cuh
          THZCReduce.cuh
          THZCReduceAll.cuh
          THZCReduceApplyUtils.cuh
          THZCAllocator.h
          THZCDeviceUtils.cuh
          THZCDeviceTensor.cuh
          THZCDeviceTensor-inl.cuh
          THZCDeviceTensorUtils.cuh
          THZCDeviceTensorUtils-inl.cuh
          DESTINATION "${THZC_INSTALL_INCLUDE_SUBDIR}/THC")
