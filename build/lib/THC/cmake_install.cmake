# Install script for directory: /home/philipp/projects/zcutorch/lib/THC

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
         RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/philipp/projects/zcutorch/build/lib/THC/libTHZC.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so"
         OLD_RPATH "/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib:/usr/local/cuda/lib64:/opt/OpenBLAS/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libTHZC.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/THC" TYPE FILE FILES
    "/home/philipp/projects/zcutorch/lib/THC/THZC.h"
    "/home/philipp/projects/zcutorch/build/lib/THC/THZCGeneral.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCGeneral.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCBlas.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCStorage.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCStorageCopy.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCTensor.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCTensorCopy.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCTensorMath.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCTensorConv.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCApply.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCReduce.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCReduceAll.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCReduceApplyUtils.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCAllocator.h"
    "/home/philipp/projects/zcutorch/lib/THC/THZCDeviceUtils.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCDeviceTensor.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCDeviceTensor-inl.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCDeviceTensorUtils.cuh"
    "/home/philipp/projects/zcutorch/lib/THC/THZCDeviceTensorUtils-inl.cuh"
    )
endif()

