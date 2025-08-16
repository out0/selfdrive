# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.20)

set(CMAKE_CUDA_ARCHITECTURES "native")
SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(gpd LANGUAGES CXX CUDA VERSION 1.0.0 DESCRIPTION "Local goal point discover lib")

include(GNUInstallDirs)

find_package(Threads REQUIRED) 
option(CUDA_LIB_JETSON "true" OFF)

set(CMAKE_CXX_STANDARD 20)

find_package(PkgConfig REQUIRED)

include(FindCUDA/select_compute_arch)
find_package(CUDA REQUIRED)

find_package(driveless REQUIRED)

# framework lib
# find_library(FRAMEWORK_LIB REQUIRED
#     NAMES driveless
#     HINTS /usr/local/lib/driveless)


if(CUDA_LIB_JETSON)
find_package(jetson-utils REQUIRED)
endif(CUDA_LIB_JETSON)

find_package(GSL REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})
include_directories("${CUDA_INCLUDE_DIRS}")
link_directories(/usr/local/lib/driveless)

include(FindPackageHandleStandardArgs)

EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
message( STATUS "Architecture: ${ARCHITECTURE}" )

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 

if (CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler found at: ${CMAKE_CUDA_COMPILER}")
    # Additional CUDA-specific actions can be performed here
else()
    message(STATUS "CUDA compiler not found or CUDA not enabled.")
endif()


# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


SET(TARGET_H
  

)

add_library(
  ${CMAKE_PROJECT_NAME} SHARED
  src/gpd.cu  
  src/gpd_pre_compute.cu  
  src/gpd_python_bind.cpp
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(CUDA_LIB_JETSON)
    target_link_libraries( ${PROJECT_NAME}
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            ${CUDA_FRAME}
            ${FRAMEWORK_LIB}
            GSL::gsl 
            GSL::gslcblas
            jetson-utils
            driveless)
else()
    target_link_libraries( ${PROJECT_NAME} 
            ${CMAKE_THREAD_LIBS_INIT}
            ${CUDA_LIBRARIES}
            ${CUDA_FRAME}
            ${FRAMEWORK_LIB}
            GSL::gsl 
            GSL::gslcblas
            cuda
            driveless
    )
endif(CUDA_LIB_JETSON)

set_target_properties( ${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/driveless
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    
 install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/driveless)


include(CMakePackageConfigHelpers)


 # Install the export set for use by find_package()
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

 # Generate and install the config and version files
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