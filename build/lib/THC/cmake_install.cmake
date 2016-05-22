# Install script for directory: /home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/philipp/torch/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so"
         RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/usr/local/magma/lib:/usr/local/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/philipp/projects/torchdistro/extra/z-cutorch/build/lib/THC/libTHZC.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so"
         OLD_RPATH "/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/usr/local/magma/lib:/usr/local/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/usr/local/magma/lib:/usr/local/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/THC" TYPE FILE FILES
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZC.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/build/lib/THC/THZCGeneral.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/FFT.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCGeneral.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCBlas.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCStorage.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCStorageCopy.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCTensor.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCTensorCopy.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCTensorMath.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCTensorConv.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCApply.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCReduce.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCReduceAll.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCReduceApplyUtils.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCAllocator.h"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCDeviceUtils.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCDeviceTensor.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCDeviceTensor-inl.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCDeviceTensorUtils.cuh"
    "/home/philipp/projects/torchdistro/extra/z-cutorch/lib/THC/THZCDeviceTensorUtils-inl.cuh"
    )
endif()

