cmake_minimum_required(VERSION 3.16)

add_definitions(-std=c++14)
project(mpJDQMR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#set(CMAKE_CXX_FLAGS "-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-o2")

set(MKLROOT ls /opt/intel/oneapi/mkl/latest/)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)

