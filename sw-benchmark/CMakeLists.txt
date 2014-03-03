nvbio_module(sw-benchmark)

addsources(
sw-benchmark.cu
)

cuda_add_executable(sw-benchmark ${sw-benchmark_srcs})
target_link_libraries(sw-benchmark nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})

