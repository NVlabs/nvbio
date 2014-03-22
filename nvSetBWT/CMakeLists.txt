nvbio_module(nvSetBWT)

addsources(
nvSetBWT.cu
)

cuda_add_executable(nvSetBWT ${nvSetBWT_srcs})
target_link_libraries(nvSetBWT nvbio moderngpu zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})

