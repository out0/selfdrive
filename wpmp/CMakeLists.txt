cmake_minimum_required(VERSION 3.28)

project(wpmp LANGUAGES CXX VERSION 0.1.0 DESCRIPTION "libwpmp")

find_package(driveless REQUIRED)
get_target_property(_driveless_defs driveless::driveless INTERFACE_COMPILE_DEFINITIONS)
if(_driveless_defs MATCHES "DRIVELESS_CUDA_ENABLED")
    message(STATUS "driveless was built with CUDA support")
    set(DRIVELESS_CUDA_ENABLED ON)
else()
    message(WARNING "driveless was built without CUDA support")
    set(DRIVELESS_CUDA_ENABLED OFF)
endif()


if (DRIVELESS_CUDA_ENABLED)
    # Optional: explicitly set NVCC if needed
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc CACHE FILEPATH "Path to nvcc")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES all-major)
    find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES all-major)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(GNUInstallDirs)

find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(GSL REQUIRED)

if (DRIVELESS_CUDA_ENABLED)
    find_package(CUDAToolkit REQUIRED)
endif()


if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
    message(STATUS "CUDA Toolkit version is >= 13.0 (found ${CUDAToolkit_VERSION})")
endif()




if (DRIVELESS_CUDA_ENABLED)
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
link_directories(/usr/local/lib/driveless)
# Library sources

if (DRIVELESS_CUDA_ENABLED)
    add_library(
    ${CMAKE_PROJECT_NAME} SHARED
    src/curve.cu
    src/collision_check.cu
    src/wave_process.cu
    src/cuda/cuda_wave_process.cu
    src/interpolation.cpp
    )
else()
    set_source_files_properties(${CMAKE_PROJECT_NAME}
        src/curve.cu
        src/collision_check.cu
        src/wave_process.cu        
        PROPERTIES LANGUAGE CXX)

   add_library(
    ${CMAKE_PROJECT_NAME} SHARED
    src/curve.cu
    src/collision_check.cu
    src/wave_process.cu
    src/interpolation.cpp
    src/cpu/cpu_wave_process.cpp
    
    )

endif()
# Properties for CUDA shared lib
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    CUDA_RUNTIME_LIBRARY Shared
)

# Include paths
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${CUDAToolkit_INCLUDE_DIRS}
)

# if(DRIVELESS_CUDA_ENABLED)
#     target_compile_definitions(${PROJECT_NAME} PRIVATE DRIVELESS_CUDA_ENABLED)
    
# endif()

# Link dependencies
if(CUDA_LIB_JETSON)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            jetson-utils
            driveless
            #CUDA::cudart
    )
else()
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            Threads::Threads
            GSL::gsl
            GSL::gslcblas
            driveless
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




