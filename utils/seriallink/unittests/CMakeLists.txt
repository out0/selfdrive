# require CMake 2.8 or greater
cmake_minimum_required(VERSION 3.18)

project(unittest)

set(CMAKE_CXX_STANDARD 17)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/include/seriallink)

find_package(PkgConfig REQUIRED)
pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)
pkg_search_module(GLIB REQUIRED glib-2.0) 

# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})


SET(TARGET_H
    ../serial_link_arduino/serial_protocol.h
    ../serial_link_arduino/seriallink.h
)

add_executable(
  unittest
  unittests.cpp
  serial_protocol_test.cpp
  serial_link_test.cpp
  ../serial_protocol.cpp
  ../seriallink.cpp
 )

# link my-recognition to jetson-inference library
target_link_libraries(unittest 
  PkgConfig::deps 
  ${GTEST_LIBRARIES} 
)