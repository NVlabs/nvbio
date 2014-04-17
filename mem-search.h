#ifndef __MEM_H
#define __MEM_H

#include <nvbio/io/fmi.h>
#include <nvbio/fmindex/mem.h>

using namespace nvbio;

struct mem_state
{
    typedef io::FMIndexDataDevice::fm_index_type    fm_index_type;
    typedef MEMFilterDevice<fm_index_type>          mem_filter_type;
    typedef io::FMIndexDataDevice::stream_type      genome_type;

    nvbio::io::FMIndexData *fmindex_data_host;
    nvbio::io::FMIndexDataDevice *fmindex_data_device;

    fm_index_type f_index, r_index;
    mem_filter_type mem_filter;
};

void mem_init(struct pipeline_context *pipeline);
void mem_search(struct pipeline_context *pipeline, const io::ReadDataDevice *batch);

#endif // ifndef __MEM_H
