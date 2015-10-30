# Install script for directory: /home/philipp/projects/zcutorch

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
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so"
         RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib" TYPE MODULE FILES "/home/philipp/projects/zcutorch/build/libzcutorch.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so"
         OLD_RPATH "/home/philipp/torch/install/lib:::::::::::::::"
         NEW_RPATH "$ORIGIN/../lib:/home/philipp/torch/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lib/libzcutorch.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/zcutorch/scm-1/lua/zcutorch" TYPE FILE FILES
    "/home/philipp/projects/zcutorch/init.lua"
    "/home/philipp/projects/zcutorch/Tensor.lua"
    "/home/philipp/projects/zcutorch/FFI.lua"
    "/home/philipp/projects/zcutorch/test/test.lua"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/philipp/projects/zcutorch/build/lib/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

file(WRITE "/home/philipp/projects/zcutorch/build/${CMAKE_INSTALL_MANIFEST}" "")
foreach(file ${CMAKE_INSTALL_MANIFEST_FILES})
  file(APPEND "/home/philipp/projects/zcutorch/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
endforeach()
