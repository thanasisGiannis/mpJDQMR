cmake_minimum_required(VERSION 3.16)

project(ex_mpJDQMR)

set(SRC ex_mpJDQMR.cpp)

add_executable(ex_mpJDQMR ${SRC})
set_target_properties(ex_mpJDQMR PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

#set_target_properties(ex_mpJDQMR PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/")

message(STATUS "!! ${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(ex_mpJDQMR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../include")
target_link_directories(ex_mpJDQMR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../lib")

target_link_libraries(ex_mpJDQMR libmatrix.so)
