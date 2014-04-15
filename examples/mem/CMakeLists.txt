nvbio_module(mem)

addsources(
mem.cu
)

cuda_add_executable(mem ${mem_srcs})
target_link_libraries(mem nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})
