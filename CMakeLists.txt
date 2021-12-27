cmake_minimum_required(VERSION 3.16)

add_definitions(-std=c++11)
project(mpJDQMR)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/matrix")
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/JDQMR)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test)

