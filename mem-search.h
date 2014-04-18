#ifndef __MEM_H
#define __MEM_H

#include <nvbio/io/fmi.h>
#include <nvbio/fmindex/mem.h>

using namespace nvbio;

struct mem_state
{
    typedef io::FMIndexDataDevice::fm_index_type                 fm_index_type;
    typedef MEMFilterDevice<fm_index_type>                       mem_filter_type;
    typedef nvbio::vector<device_tag, mem_filter_type::mem_type> mem_vector_type;
    typedef io::FMIndexDataDevice::stream_type                   genome_type;

    nvbio::io::FMIndexData *fmindex_data_host;
    nvbio::io::FMIndexDataDevice *fmindex_data_device;

    // the FM-index objects
    fm_index_type f_index, r_index;

    // our MEM filter object, used to rank and locate MEMs and keep track of statistics
    mem_filter_type mem_filter;
    // the result vector for mem_search
    mem_vector_type mems;
};

void mem_init(struct pipeline_context *pipeline);
void mem_search(struct pipeline_context *pipeline, const io::ReadDataDevice *batch);
void mem_split(struct pipeline_context *pipeline, const io::ReadDataDevice *batch);

#endif // ifndef __MEM_H
