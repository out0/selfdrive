cmake_minimum_required(VERSION 3.26)


set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(driveless-cuda-rrt-accel LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "CUDA Compute for OG")

include(GNUInstallDirs)
option(CUDA_LIB_JETSON "true" OFF)
find_package(Threads REQUIRED)

include(FindCUDA/select_compute_arch)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")


if(CUDA_LIB_JETSON)
    find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)

include(FindPackageHandleStandardArgs)


SET(TARGET_H
    # cuda_frame.h
    # cuda_basic.h
    # class_def.h
)

#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -dc")

add_library( ${CMAKE_PROJECT_NAME} SHARED
    src/driveless_cuda_graph.cu
    src/cuda_graph_optim.cu
    src/python_bind.cpp
    src/driveless_cuda_graph.cpp
)

if(CUDA_LIB_JETSON)
    target_link_libraries( ${PROJECT_NAME}
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            jetson-utils)
else()
    target_link_libraries( ${PROJECT_NAME} 
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            cuda
    )
endif(CUDA_LIB_JETSON)

set_target_properties(  ${CMAKE_PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
install(TARGETS  ${CMAKE_PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    