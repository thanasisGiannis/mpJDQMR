cmake_minimum_required(VERSION 3.16)

add_definitions(-std=c++14)
project(mpJDQMR)

set(MKLROOT ls /opt/intel/oneapi/mkl/latest/)
find_package(MKL CONFIG REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)

