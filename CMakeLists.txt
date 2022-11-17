cmake_minimum_required(VERSION 3.19)
project(oclcg C)

set(CMAKE_C_STANDARD 99)

find_package(OpenCL REQUIRED)

include_directories(
        "./bebop/bebop_util/include"
        "./bebop/sparse_matrix_converter/include"
)
find_library(BEBOP bebop_util PATHS ./bebop/bebop_util)
find_library(SMC sparse_matrix_converter PATHS ./bebop/sparse_matrix_converter)

add_library(oclcg SHARED main.c clcg.h clcg.c )
add_executable(oclcgex main.c clcg.h clcg.c)

target_link_libraries(oclcg OpenCL::OpenCL m "${BEBOP}" "${SMC}")
target_link_libraries(oclcgex OpenCL::OpenCL m "${BEBOP}" "${SMC}")
