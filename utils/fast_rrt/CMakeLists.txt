# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(fastrrt LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "libfastrrt")

include(GNUInstallDirs)

find_package(Threads REQUIRED) 
option(CUDA_LIB_JETSON "true" OFF)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)

include(FindCUDA/select_compute_arch)
find_package(CUDA REQUIRED)

if(CUDA_LIB_JETSON)
find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)

find_library(LIB_CUDA_UTILS REQUIRED
    NAMES cuda_utils
    HINTS /usr/local/lib/driveless
)

find_library(CUDA_FRAME REQUIRED
    NAMES driveless-cudac
    HINTS /usr/lib
)


include_directories(${PROJECT_SOURCE_DIR})
include_directories("${CUDA_INCLUDE_DIRS}")

include(FindPackageHandleStandardArgs)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 


# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


SET(TARGET_H
  

)

add_library(
  ${CMAKE_PROJECT_NAME} SHARED
  src/cuda/kinematics.cu
  src/cuda/path_feasible.cu
  src/cuda/random_number_gen.cu
  src/cuda/graph.cu
  src/cuda/graph_list.cu
  src/cuda/graph_best_node.cu
  src/cuda/graph_optimization.cu
  src/cuda/graph_node_derivation.cu
  src/cuda/graph_costs.cu
  src/cuda/cuda_grid.cu
  src/cuda/apf.cu
#  src/cuda/graph_clear_except.cu
  src/fastrrt.cpp
  src/angle.cpp
  src/waypoint.cpp
  src/python_bind.cpp
  src/python_cuda_graph_bind.cpp
  src/interpolation.cpp
  #src/debug.cpp
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(CUDA_LIB_JETSON)
    target_link_libraries( ${PROJECT_NAME}
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            ${CUDA_FRAME}
            ${LIB_CUDA_UTILS}
            jetson-utils)
else()
    target_link_libraries( ${PROJECT_NAME} 
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            ${CUDA_FRAME}
            ${LIB_CUDA_UTILS}
            cuda
    )
endif(CUDA_LIB_JETSON)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    W no-deprecated-gpu-targets
)


install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless)
    
 install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless)
