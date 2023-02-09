cmake_minimum_required(VERSION 3.16)

add_definitions(-std=c++2a)
project(mpJDQMR)

if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
endif()


#set(CMAKE_C_COMPILER gcc-10)
#set(CMAKE_CXX_COMPILER g++-10)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g3")
#set(CMAKE_CXX_FLAGS_RELEASE "-o3")

set(MKLROOT ls /opt/intel/oneapi/mkl/latest/)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/examples)

