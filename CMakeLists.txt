cmake_minimum_required(VERSION 3.20)
project(sparse_linalg VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SPARSE_LINALG_BUILD_TESTS "Build tests" ON)
option(SPARSE_LINALG_BUILD_EXAMPLES "Build examples" ON)
option(SPARSE_LINALG_BUILD_BENCHMARKS "Build benchmarks" ON)
option(SPARSE_LINALG_ENABLE_SANITIZERS "Enable sanitizers in Debug mode" ON)

add_library(sparse_linalg INTERFACE)
target_include_directories(sparse_linalg INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_options(sparse_linalg INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wconversion
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Woverloaded-virtual
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND SPARSE_LINALG_ENABLE_SANITIZERS)
    message(STATUS "Enabling sanitizers for Debug build")
    target_compile_options(sparse_linalg INTERFACE
        -fsanitize=address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(sparse_linalg INTERFACE
        -fsanitize=address,undefined
    )
endif()

if(SPARSE_LINALG_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(SPARSE_LINALG_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(SPARSE_LINALG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(GNUInstallDirs)
install(TARGETS sparse_linalg
    EXPORT sparse_linalg-targets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT sparse_linalg-targets
    FILE sparse_linalg-targets.cmake
    NAMESPACE sparse_linalg::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/sparse_linalg
)