# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.18)

project(tst)

set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/include/seriallink)

find_package(PkgConfig REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 

find_library(SERIAL_LINK REQUIRED
    NAMES seriallink
    HINTS /usr/lib
)


add_executable(
  tst
  tst.cpp
 )

# link my-recognition to jetson-inference library
target_link_libraries(tst 
  PkgConfig::deps 
  ${SERIAL_LINK} 
)