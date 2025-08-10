# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

project(unittest LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "libgdp test")
include(GNUInstallDirs)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

find_package(Threads REQUIRED) 

message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

set(CMAKE_CXX_STANDARD 20)
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
link_directories(/usr/local/lib/driveless)
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


SET(TARGET_H
  

)

find_package(driveless REQUIRED)

add_executable(
  ${CMAKE_PROJECT_NAME} 
  src/unittests.cpp
  src/test_utils.cpp
  #src/test_gpd_with_heading.cpp
  #src/test_gpd_with_heading_exclusion_zone.cpp
  #src/test_gpd_lowest_cost_to_goal.cpp
  src/test_gpd_lowest_cost_to_goal_exclusion_zone.cpp
  #-------------
  ../../src/gpd.cu
  ../../src/gpd_pre_compute.cu
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories( ${CMAKE_PROJECT_NAME}  PRIVATE ${Python3_INCLUDE_DIRS})


if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options( ${CMAKE_PROJECT_NAME}  PUBLIC "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
  target_link_libraries( ${CMAKE_PROJECT_NAME}  "${CMAKE_THREAD_LIBS_INIT}")
endif()



# link my-recognition to jetson-inference library
target_link_libraries( 
  ${CMAKE_PROJECT_NAME}  
  PkgConfig::deps 
  ${GTEST_LIBRARIES} 
  ${CUDA_LIBRARIES}
  ${CUDA_FRAME}
  ${OpenCV_LIBS}
  driveless
  # ${Python3_LIBRARIES}
)

