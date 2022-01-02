cmake_minimum_required(VERSION 3.16)

project(lib_mpJDQMR)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/blasWrappers/blasWrappers.cpp
#				${CMAKE_CURRENT_SOURCE_DIR}/jd/*
#				${CMAKE_CURRENT_SOURCE_DIR}/matrix/*
#				${CMAKE_CURRENT_SOURCE_DIR}/sqmr/*
)

add_library(lib_mpJDQMR SHARED ${SRC})


target_include_directories(lib_mpJDQMR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

target_link_directories(lib_mpJDQMR PUBLIC "${MKLROOT}/lib/intel64")
target_include_directories(lib_mpJDQMR PUBLIC "-DMKL_ILP64 -m64 ${MKLROOT}/include/")

set_target_properties(lib_mpJDQMR PROPERTIES OUTPUT_NAME mpjdqmr)
set_target_properties(lib_mpJDQMR PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../lib)

target_link_libraries(lib_mpJDQMR 
											-lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core 
											-lmkl_blacs_intelmpi_ilp64 -lgomp -lpthread -lm -ldl
										)
