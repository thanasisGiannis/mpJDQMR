cmake_minimum_required(VERSION 3.16)

project(matrix )

set(SRC matrix.tcc)

add_library(matrix SHARED ${SRC})
set_target_properties(matrix PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(matrix PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../lib")
target_include_directories(matrix PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../../include")


