# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

project(unittest LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "CUDA Compute for OG Unittest")
include(GNUInstallDirs)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(Threads REQUIRED) 

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR})

find_package(PkgConfig REQUIRED)
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

find_library(CUDA_FRAME_LIB REQUIRED
    NAMES driveless-cudac
    HINTS /usr/lib
)



EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 


# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})


SET(TARGET_H
  

)



add_executable(
  unittest
  unittests.cpp
  test_cuda_accel_rrt.cpp
  ../../src/driveless_cuda_graph.cu
  ../../src/driveless_cuda_graph.cpp
  ../../src/python_bind.cpp
)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(unittest PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries(unittest "${CMAKE_THREAD_LIBS_INIT}")
endif()

# link my-recognition to jetson-inference library
target_link_libraries(unittest PkgConfig::deps 
  ${GTEST_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${CUDA_FRAME_LIB}
)