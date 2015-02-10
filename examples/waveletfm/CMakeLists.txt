nvbio_module(waveletfm)

addsources(
waveletfm.cu
)

cuda_add_executable(waveletfm ${waveletfm_srcs})
target_link_libraries(waveletfm nvbio moderngpu zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})
