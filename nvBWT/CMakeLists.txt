nvbio_module(nvBWT)

addsources(
nvBWT.cpp
filelist.cpp
)

# figure out the name for libbwtsw
if ("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
  set(bwtsw_lib "${CMAKE_SOURCE_DIR}/contrib/libbwtsw-bin/libbwtsw-linux-amd64-release.a")
endif()

if (WIN32)
  set(bwtsw_lib "${CMAKE_SOURCE_DIR}/contrib/libbwtsw-bin/bwtsw.lib")
endif()

cuda_add_executable(nvBWT ${nvBWT_srcs})
target_link_libraries(nvBWT nvbio ${bwtsw_lib} crcstatic ${SYSTEM_LINK_LIBRARIES})

