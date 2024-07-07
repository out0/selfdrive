cmake_minimum_required(VERSION 3.11)
project(datalink VERSION 1.0.0 DESCRIPTION "datalink")

include(GNUInstallDirs)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED) 

include(FindPackageHandleStandardArgs)

pkg_check_modules(deps REQUIRED IMPORTED_TARGET glib-2.0)


SET(TARGET_H
    datalink.h
    databridge.h
)
 
add_library(datalink SHARED
    datalink.cpp
    datalink_connection.cpp
    datalink_server_connection.cpp
    datalink_client_connection.cpp
    datalink_protocol.cpp
    databridge.cpp
    pybind/pybind.cpp
)

target_link_libraries(datalink 
        PkgConfig::deps 
        ${CMAKE_THREAD_LIBS_INIT}
        ${GLIB_LDFLAGS}
       
)

add_definitions(-D __DATALINK_COMPILE=1)

set_target_properties(datalink PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "datalink.h;databridge.h;datalink_result.h"
)

install(TARGETS datalink
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION "/usr/include")