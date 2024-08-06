cmake_minimum_required(VERSION 3.24)
project(driveless-cudac LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "CUDA Compute for OG")

include(GNUInstallDirs)
option(CUDA_LIB_JETSON "true" OFF)
find_package(Threads REQUIRED)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(CUDA_LIB_JETSON)
    find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)

include(FindPackageHandleStandardArgs)

SET(CUDACXX /usr/local/cuda-12.2/bin/nvcc)

SET(TARGET_H
    # cuda_frame.h
    # cuda_basic.h
    # class_def.h
)
 
add_library(driveless-cudac SHARED
    driveless_cuda_frame.cpp
    driveless_cuda_frame.cu
    python_bind.cpp
)

if(CUDA_LIB_JETSON)
    target_link_libraries(driveless-cudac 
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            jetson-utils)
else()
    target_link_libraries(driveless-cudac 
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            cuda
    )
endif(CUDA_LIB_JETSON)

set_target_properties(driveless-cudac PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
install(TARGETS driveless-cudac
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    