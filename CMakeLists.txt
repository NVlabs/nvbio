project(nvBio)

cmake_minimum_required(VERSION 2.8)

# available options
option(CUDA_VERBOSE_PTXAS
  "Enable verbose output from PTXAS"
  ON)

option(PROFILING
  "Enable profiling"
  OFF)

option(WERROR
  "Treat compiler warnings as errors"
  OFF)

# import our own nvbio_* macros
include("${CMAKE_SOURCE_DIR}/cmake-local/nvbio.cmake")
# grab gcc flags
include("${CMAKE_SOURCE_DIR}/cmake-local/gcc.cmake")

# default to debug builds
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Building ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
  add_definitions(-DTHRUST_DEBUG)
  add_definitions(-DCUDA_DEBUG)
  set(CUDA_NVCC_DEBUG_FLAGS "-g -lineinfo")
else()
  set(CUDA_NVCC_DEBUG_FLAGS "")
endif()

# path to install binaries to
# this is the same place that the old Linux makefiles placed the binaries in
if (NOT WIN32)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
endif()

find_package(CUDA)
find_package(Doxygen)

# set up required system libs
set(SYSTEM_LINK_LIBRARIES "")
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  list(APPEND SYSTEM_LINK_LIBRARIES gomp)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  # shm_* live in librt on Linux
  list(APPEND SYSTEM_LINK_LIBRARIES rt)
endif()

if (MSVC_IDE)
  # suppress automatic regeneration of VS project files
  #set(CMAKE_SUPPRESS_REGENERATION ON)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_NVCC_DEBUG_FLAGS} --gpu-architecture=sm_35 -use_fast_math -Xcompiler -fopenmp)

set(cudpp_INCLUDE_DIRS ${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/contrib ${CMAKE_BINARY_DIR}/contrib/zlib-1.2.7 ${CMAKE_SOURCE_DIR}/contrib/crc ${CMAKE_SOURCE_DIR}/contrib/bamtools ${CMAKE_SOURCE_DIR}/contrib/moderngpu/include)

add_subdirectory(contrib/zlib-1.2.7)
add_subdirectory(contrib/crc)
add_subdirectory(contrib/bamtools)
add_subdirectory(contrib/moderngpu)

add_subdirectory(nvbio)
add_subdirectory(nvBowtie)
add_subdirectory(nvFM-server)
add_subdirectory(nvBWT)
add_subdirectory(nvSSA)
add_subdirectory(nvbio-test)
add_subdirectory(nvbio-aln-diff)
add_subdirectory(sw-benchmark)

nvbio_module(nvbio-project)
nvbio_doxygen()

