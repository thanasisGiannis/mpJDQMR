cmake_minimum_required(VERSION 3.16)

add_definitions(-std=c++2a)
project(mpJDQMR)


if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
endif()


set(CMAKE_CXX_FLAGS "-O3 -g3 -mavx -ftree-vectorizer-verbose=2 -Wall -Wextra -ftree-vectorize")
#set(CMAKE_CXX_FLAGS_DEBUG "-g3")
#set(CMAKE_CXX_FLAGS_RELEASE "-o3")

set(MKLROOT ls /opt/intel/oneapi/mkl/latest/)
find_package(MKL CONFIG REQUIRED)
find_package(OpenMP)

#find_package(icpc REQUIRED)

#set directory variables for each sub-component
set(MPJDQMRDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(BLASWRAPPERSDIR ${MPJDQMRDIR}/src/blasWrappers)
set(TESTSDIR ${MPJDQMRDIR}/tests)
set(MPJDQMRINCLUDE ${MPJDQMRDIR}/include)

add_subdirectory(${MPJDQMRDIR}/src)
add_subdirectory(${MPJDQMRDIR}/examples)
add_subdirectory(${MPJDQMRDIR}/tests)

set(CMAKE_C_COMPILER /opt/intel/oneapi/compiler/latest/linux/bin/icx)
set(CMAKE_CXX_COMPILER /opt/intel/oneapi/compiler/latest/linux/bin/icpx)

