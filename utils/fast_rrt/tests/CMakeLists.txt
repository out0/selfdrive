# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

project(unittest LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "libfastrrt test")
include(GNUInstallDirs)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(Threads REQUIRED) 


# libdriveless
find_library(CUDA_FRAME REQUIRED
    NAMES driveless-cudac
    HINTS /usr/lib
)

find_library(LIB_CUDA_UTILS REQUIRED
    NAMES cuda_utils
    HINTS /usr/local/lib/driveless
)

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CUDA_FLAGS_DEBUG 1)
include_directories(${PROJECT_SOURCE_DIR})

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 


# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -W no-deprecated-gpu-targets")

SET(TARGET_H
  

)

add_executable(
  ${CMAKE_PROJECT_NAME} 
  ../src/cuda/kinematics.cu
  ../src/cuda/path_feasible.cu
  ../src/cuda/random_number_gen.cu
  ../src/cuda/graph.cu
  ../src/cuda/graph_list.cu
  ../src/cuda/graph_node_derivation.cu
  ../src/cuda/graph_best_node.cu
  ../src/cuda/graph_nodes_in_range.cu
  ../src/cuda/graph_optimization.cu
  ../src/cuda/graph_costs.cu
  ../src/cuda/cuda_grid.cu
  ../src/cuda/apf.cu
  #../src/cuda/graph_clear_except.cu
  ../src/fastrrt.cpp
  ../src/angle.cpp
  ../src/waypoint.cpp
  ../src/python_bind.cpp
  ../src/python_cuda_graph_bind.cpp
  ../src/interpolation.cpp
  #../src/star_optim.cpp

  #../src/debug.cpp
  src/test_utils.cpp
  src/unittests.cpp
  src/test_graph.cpp
  src/test_kinematics.cpp
  src/test_graph_node_derivation.cpp
  src/test_graph_best_node.cpp
  src/test_rrt.cpp
  #src/test_optimize_path.cpp
  src/test_simple_apf.cpp
)

set_property(
  TARGET ${CMAKE_PROJECT_NAME} 
  PROPERTY CUDA_SEPARABLE_COMPILATION ON )

target_include_directories( ${CMAKE_PROJECT_NAME}  PRIVATE ${Python3_INCLUDE_DIRS})


if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options( ${CMAKE_PROJECT_NAME}  PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries( ${CMAKE_PROJECT_NAME}  "${CMAKE_THREAD_LIBS_INIT}" )
endif()

# link my-recognition to jetson-inference library
target_link_libraries( 
  ${CMAKE_PROJECT_NAME}  
  PkgConfig::deps 
  ${GTEST_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${CUDA_FRAME}
  ${OpenCV_LIBS}
  ${CUDA_FRAME}
  # ${Python3_LIBRARIES}
  ${LIB_CUDA_UTILS}

)

