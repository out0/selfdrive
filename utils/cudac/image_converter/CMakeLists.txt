cmake_minimum_required(VERSION 3.11)
project(bevconv LANGUAGES CXX CUDA VERSION 1.0.0 DESCRIPTION "CUDA Compute for OG testing")

include(GNUInstallDirs)

find_package(Threads REQUIRED) 
find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

if(CUDA_LIB_JETSON)
    find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)


include(FindPackageHandleStandardArgs)
include_directories(${OpenCV_INCLUDE_DIRS})


SET(TARGET_H
    driveless_cudac.h
)
 
add_executable(${PROJECT_NAME}
    img_conv.cpp
    ../src/frame_color.cu
    ../src/driveless_cuda_frame.cpp
    ../src/frame_process.cu
)

if(CUDA_LIB_JETSON)
    target_link_libraries(${PROJECT_NAME} 
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_LIBRARIES}
        jetson-utils
        ${OpenCV_LIBS})
else()
    target_link_libraries(${PROJECT_NAME} 
        ${CMAKE_THREAD_LIBS_INIT}
        ${CUDA_LIBRARIES}
        cuda
        ${OpenCV_LIBS})

endif(CUDA_LIB_JETSON)


set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

    