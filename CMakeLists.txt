cmake_minimum_required(VERSION 3.18)

project(sismo2 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
    main.cpp
    EDO.cpp
    variables.cpp
)

set(HEADERS
    EDO.hpp
    variables.hpp
)

set(SUNDIALS_DIR "F:/stage/sundials_install")

find_package(SUNDIALS REQUIRED COMPONENTS cvode nvecserial)

add_executable(sismo2 ${SOURCES} ${HEADERS})

target_include_directories(sismo2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(sismo2 PUBLIC SUNDIALS::cvode SUNDIALS::nvecserial)