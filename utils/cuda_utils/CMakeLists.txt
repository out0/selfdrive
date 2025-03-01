# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cuda_utils LANGUAGES CXX CUDA VERSION 2.0.0 DESCRIPTION "CUDA basic utitilites that connect Jetson and PC")

include(GNUInstallDirs)

# find_package(Threads REQUIRED) 
option(CUDA_LIB_JETSON "true" OFF)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)

include(FindCUDA/select_compute_arch)
find_package(CUDA REQUIRED)

if(CUDA_LIB_JETSON)
find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)


include_directories(${PROJECT_SOURCE_DIR})
include_directories("${CUDA_INCLUDE_DIRS}")

include(FindPackageHandleStandardArgs)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


SET(TARGET_H
  

)

add_library(
  ${CMAKE_PROJECT_NAME} SHARED
  src/cuda_utils.cpp
)

if(CUDA_LIB_JETSON)
    target_link_libraries( ${PROJECT_NAME}
            ${CUDA_LIBRARIES}
            jetson-utils)
else()
    target_link_libraries( ${PROJECT_NAME} 
            ${CUDA_LIBRARIES}
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
