cmake_minimum_required(VERSION 3.26)

project(test_executor LANGUAGES CXX CUDA VERSION 1.0.0 DESCRIPTION "C++ port of test_executor.py")

set(CMAKE_CXX_STANDARD 20)

include(GNUInstallDirs)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)   # or drop this and keep the vendored json.hpp next to this file
find_package(driveless REQUIRED)
find_package(fastrrt REQUIRED)

include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})   # so "#include <nlohmann/json.hpp>" / test_utils.h resolve

link_directories(/usr/local/lib/driveless)

add_executable(${CMAKE_PROJECT_NAME}
  test_executor.cpp
)

if(THREADS_HAVE_PTHREAD_ARG)
  target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC "-pthread")
endif()

target_link_libraries(${CMAKE_PROJECT_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
  ${CUDA_LIBRARIES}
  ${OpenCV_LIBS}
  driveless
  fastrrt
)
