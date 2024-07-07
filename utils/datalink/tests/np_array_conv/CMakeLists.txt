cmake_minimum_required(VERSION 3.11)
project(simplelib VERSION 0.0.1 DESCRIPTION "simple test lib")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(GTest REQUIRED)

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)


SET(TARGET_H
)
 
add_library(simplelib SHARED
    simple_lib.cpp
)

add_executable(
  unittest
  simple_lib.cpp
  tst_simple_lib.cpp
)
 
target_link_libraries(simplelib 
        PkgConfig::deps 
        ${GLIB_LDFLAGS}
       
)

set_target_properties(simplelib PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)

target_link_libraries(unittest 
    PkgConfig::deps 
    ${GTEST_LIBRARIES} 
  #${DATALINK_LIB}
)

install(TARGETS simplelib
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

