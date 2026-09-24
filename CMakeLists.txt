cmake_minimum_required(VERSION 3.21)
project(parakeet_redux
    VERSION 1.0.0
    DESCRIPTION "Dependency-free C inference for parakeet-redux (1.58-bit parakeet-tdt-0.6b-v3)"
    HOMEPAGE_URL "https://github.com/eschmidbauer/parakeet-redux-c"
    LANGUAGES C)

# Standalone builds also produce the CLI and the concurrency test. Consumers
# that add this directory (FetchContent, add_subdirectory) get the library only.
option(PARAKEET_BUILD_CLI "Build the parakeet command-line binary" ${PROJECT_IS_TOP_LEVEL})
set(PARAKEET_SIMD "native" CACHE STRING "native (NEON/AVX2 for the target), or none for the plain-C kernels")
set(PARAKEET_MARCH "" CACHE STRING "extra -march flag; Linux aarch64 defaults to armv8.2-a+dotprod+fp16")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

file(GLOB PARAKEET_LIB_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM PARAKEET_LIB_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c)

add_library(parakeet STATIC ${PARAKEET_LIB_SOURCES})
add_library(parakeet::parakeet ALIAS parakeet)

target_include_directories(parakeet
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>)
target_compile_options(parakeet PRIVATE -O3 -fno-math-errno -Wall -Wextra -Wno-unused-parameter -Wno-unused-function)
target_compile_definitions(parakeet PRIVATE _DEFAULT_SOURCE _DARWIN_C_SOURCE)

if(PARAKEET_SIMD STREQUAL "none")
    target_compile_definitions(parakeet PRIVATE PK_GENERIC_KERNELS)
endif()
if(PARAKEET_MARCH)
    target_compile_options(parakeet PRIVATE ${PARAKEET_MARCH})
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT PARAKEET_SIMD STREQUAL "none")
    target_compile_options(parakeet PRIVATE -mavx2 -mfma -mf16c)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$" AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(parakeet PRIVATE -march=armv8.2-a+dotprod+fp16)
endif()

find_package(Threads REQUIRED)
target_link_libraries(parakeet PUBLIC Threads::Threads m)

if(PARAKEET_BUILD_CLI)
    add_executable(parakeet_cli src/main.c)
    set_target_properties(parakeet_cli PROPERTIES OUTPUT_NAME parakeet)
    target_compile_definitions(parakeet_cli PRIVATE _DEFAULT_SOURCE _DARWIN_C_SOURCE)
    target_link_libraries(parakeet_cli PRIVATE parakeet)
    add_executable(concurrent_test tests/concurrent.c)
    target_compile_definitions(concurrent_test PRIVATE _DEFAULT_SOURCE _DARWIN_C_SOURCE)
    target_link_libraries(concurrent_test PRIVATE parakeet)
endif()
