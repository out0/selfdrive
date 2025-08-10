# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

project(unittest LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "libdriveless test")
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

find_package(GSL REQUIRED)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 


# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


SET(TARGET_H
  

)

add_executable(
  ${CMAKE_PROJECT_NAME} 
  src/unittests.cpp
  src/test_utils.cpp
  #-------------
  src/test_angle.cpp
  src/test_waypoint.cpp
  src/test_map_pose.cpp
  src/test_map_pose_nearest_goal_pose.cpp
  src/test_world_pose.cpp
  src/test_async_component.cpp
  src/test_cuda_frame.cpp
  src/test_search_frame.cpp
  src/test_search_frame_set_goal.cpp
  src/test_search_frame_set_goal_vectorized.cpp
  src/test_search_frame_check_fesible_path.cpp
  src/test_search_frame_check_fesible_path_fesible.cpp
  src/test_search_frame_check_fesible_path_unfeasible.cpp
  src/test_search_frame_is_traversable.cpp
  src/test_coord_conversion_world_pose.cpp
  src/test_coord_conversion_map_pose.cpp
  src/test_coord_conversion_waypoint.cpp
  src/test_quaternion.cu
  src/test_interpolator.cpp
  ../../src/waypoint.cpp
  ../../src/angle.cpp
  ../../src/map_pose.cpp
  ../../src/world_pose.cpp
  ../../src/async_component.cpp
  ../../src/coord_conversion.cpp
  ../../src/interpolator.cpp
  ../../src/cuda/cuda_frame.cu
  ../../src/cuda/quaternion.cu
  ../../src/cuda/search_frame_color.cu
  ../../src/cuda/search_frame_moving_obstacles.cu
  ../../src/cuda/search_frame_path_feasible.cu
  ../../src/cuda/search_frame_pre_process.cu
  ../../src/cuda/search_frame.cu
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
  GSL::gsl 
  GSL::gslcblas
  # ${Python3_LIBRARIES}
)

