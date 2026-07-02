cmake_minimum_required(VERSION 3.30)

option(CUDA_ENABLED "true" ON)

if(CUDA_ENABLED)
    project(driveless LANGUAGES CXX CUDA VERSION 2.3.0 DESCRIPTION "libdriveless")
else()
    project(driveless LANGUAGES CXX VERSION 2.3.0 DESCRIPTION "libdriveless")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/driveless_config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/driveless_config.h
    @ONLY
)

set(CMAKE_CXX_STANDARD 20)

if(CUDA_ENABLED)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

# Optional: explicitly set NVCC if needed
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")

include(GNUInstallDirs)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GSL REQUIRED)

if(DRIVELESS_CUDA_ENABLED)
    find_package(CUDAToolkit REQUIRED)

    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
        message(STATUS "CUDA Toolkit version is >= 13.0 (found ${CUDAToolkit_VERSION})")
    endif()

    string(REPLACE "." ";" CUDA_VER_LIST ${CMAKE_CUDA_COMPILER_VERSION})
    list(GET CUDA_VER_LIST 0 CUDA_MAJOR)
    list(GET CUDA_VER_LIST 1 CUDA_MINOR)

    add_compile_definitions(CUDA_VERSION_MAJOR=${CUDA_MAJOR})
    add_compile_definitions(CUDA_VERSION_MINOR=${CUDA_MINOR})

endif()


option(CUDA_LIB_JETSON "true" OFF)

if(CUDA_LIB_JETSON)
    find_package(jetson-utils REQUIRED)
endif()

# Detect architecture
execute_process(
    COMMAND uname -m
    COMMAND tr -d '\n'
    OUTPUT_VARIABLE ARCHITECTURE
)
message(STATUS "Architecture: ${ARCHITECTURE}")

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0)

if(DRIVELESS_CUDA_ENABLED)
# Library sources
set_source_files_properties(
    src/quaternion.cpp
    PROPERTIES LANGUAGE CUDA)
add_library(${PROJECT_NAME} SHARED
    src/angle.cpp
    src/map_pose.cpp
    src/waypoint.cpp
    src/world_pose.cpp
    src/async_component.cpp
    src/coord_conversion.cpp
    src/interpolator.cpp
    src/search_params.cpp
    src/quaternion.cpp
    src/cuda_basic.cpp
    src/cpu_parallel_processor.cpp
    src/cuda/cuda_frame.cu
    src/cpu_parallel_copy.cpp
    src/cuda/quaternion.cu
    src/cuda/search_frame.cu
    src/cuda/search_frame_color.cu
    src/cuda/search_frame_pre_process.cu
    src/cuda/search_frame_path_feasible.cu
    src/cuda/search_frame_python_bind_cuda.cpp
    src/cuda/collision_check.cu
    src/cuda/compute_heading.cu
    src/cpu/search_frame_python_bind_cpu.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CUDA_RUNTIME_LIBRARY Shared
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDAToolkit_INCLUDE_DIRS}
)

if(CUDA_ENABLED)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DRIVELESS_CUDA_ENABLED)    
endif()


else()
set_source_files_properties(${PROJECT_NAME}
    src/cuda/collision_check.cu
    src/cuda/compute_heading.cu
    src/quaternion.cpp 
    PROPERTIES LANGUAGE CXX)
add_library(${PROJECT_NAME} SHARED
    src/angle.cpp
    src/map_pose.cpp
    src/waypoint.cpp
    src/world_pose.cpp
    src/async_component.cpp
    src/coord_conversion.cpp
    src/interpolator.cpp
    src/search_params.cpp
    src/cpu_parallel_processor.cpp
    src/cpu_parallel_copy.cpp  
    src/quaternion.cpp
    src/cuda_basic.cpp
    src/cpu/search_frame_cpu.cpp
    src/cpu/search_frame_color_cpu.cpp
    src/cpu/search_frame_pre_process_cpu.cpp
    src/cpu/search_frame_path_feasible_cpu.cpp
    src/cpu/search_frame_python_bind_cpu.cpp
    src/cuda/collision_check.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

endif()

# Link dependencies

if(CUDA_LIB_JETSON)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            jetson-utils
            #CUDA::cudart
    )
else()
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            #CUDA::cudart
    )
endif()

# Install library
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless
)

# Install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless
)

# Export set for find_package()
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/driveless_config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless
)

# Generate package config files
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)
