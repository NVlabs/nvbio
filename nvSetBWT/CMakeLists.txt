nvbio_module(nvSetBWT)

addsources(
nvSetBWT.cu
)

cuda_add_executable(nvSetBWT ${nvSetBWT_srcs})
target_link_libraries(nvSetBWT nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})

