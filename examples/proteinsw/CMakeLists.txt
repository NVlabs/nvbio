nvbio_module(proteinsw)

addsources(
proteinsw.cu
)

cuda_add_executable(proteinsw ${proteinsw_srcs})
target_link_libraries(proteinsw nvbio zlibstatic crcstatic ${SYSTEM_LINK_LIBRARIES})
