nvbio_module(proteinsw)

cuda_add_executable(proteinsw proteinsw.cu)
target_link_libraries(proteinsw nvbio zlibstatic lz4 crcstatic ${SYSTEM_LINK_LIBRARIES})

cuda_add_executable(proteinsw_qp proteinsw_qp.cu)
target_link_libraries(proteinsw_qp nvbio zlibstatic lz4 crcstatic ${SYSTEM_LINK_LIBRARIES})
