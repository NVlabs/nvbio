/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#include "mem-search.h"
#include "options.h"
#include "pipeline.h"

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

using namespace nvbio;

static io::FMIndexData *load_genome(const char *name, uint32 flags)
{
    if (command_line_options.genome_use_mmap)
    {
        // try loading via mmap first
        io::FMIndexDataMMAP *mmap_loader = new io::FMIndexDataMMAP();
        if (mmap_loader->load(name))
        {
            return mmap_loader;
        }

        delete mmap_loader;
    }

    // fall back to file name
    io::FMIndexDataRAM *file_loader = new io::FMIndexDataRAM();
    if (!file_loader->load(name, flags))
    {
        log_error(stderr, "could not load genome %s\n");
        exit(1);
    }

    return file_loader;
}

void mem_init(struct pipeline_context *pipeline)
{
    // this specifies which portions of the FM index data to load
    const uint32 fm_flags = io::FMIndexData::GENOME  |
                            io::FMIndexData::FORWARD |
                            io::FMIndexData::REVERSE |
                            io::FMIndexData::SA;

    // load the genome on the host
    pipeline->mem.fmindex_data_host = load_genome(command_line_options.genome_file_name, fm_flags);
    // copy genome data to device
    pipeline->mem.fmindex_data_device = new io::FMIndexDataDevice(*pipeline->mem.fmindex_data_host, fm_flags);

    pipeline->mem.f_index = pipeline->mem.fmindex_data_device->index();
    pipeline->mem.r_index = pipeline->mem.fmindex_data_device->rindex();
}

// search MEMs for all reads in batch
void mem_search(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;
    const uint32 n_reads = batch->size();

    // reset the filter
    mem->mem_filter = mem_state::mem_filter_type();

    // search for MEMs
    mem->mem_filter.rank(THRESHOLD_KMEM_SEARCH, mem->f_index, mem->r_index, batch->const_read_string_set(),
                         command_line_options.min_intv, command_line_options.max_intv, command_line_options.min_span);

    log_info(stderr, "%.1f average ranges\n", float(mem->mem_filter.n_ranges()) / float(n_reads));
    log_info(stderr, "%.1f average MEMs\n", float(mem->mem_filter.n_mems()) / float(n_reads));
}

namespace {

#if defined(NVBIO_DEVICE_COMPILATION)
  #define HOST_DEVICE_STATEMENT( host, device ) device
#else
  #define HOST_DEVICE_STATEMENT( host, device ) host
#endif

// a functor to count the number of MEM hits produced by a given range of reads
struct hit_count_functor
{
    typedef uint32 argument_type;
    typedef uint32 result_type;

    // constructor
    hit_count_functor(struct mem_state* _mem) : mem( _mem ) {}

    // return the number of hits up to a given read
    NVBIO_HOST_DEVICE                                   // silence nvcc - this function is host only
    uint32 operator() (const uint32 read_id) const
    {
        return HOST_DEVICE_STATEMENT( mem->mem_filter.first_hit( read_id ), 0u );
    }

    struct mem_state* mem;
};

};

// given the first read in a chunk, determine a suitably sized chunk of reads
// (for which we can locate all MEMs in one go), updating pipeline::chunk
void fit_read_chunk(
    struct pipeline_context     *pipeline,
    const io::ReadDataDevice    *batch,
    const uint32                read_begin)     // first read in the chunk
{
    struct mem_state *mem = &pipeline->mem;

    const uint32 max_hits = command_line_options.mems_batch;

    //
    // use a binary search to locate the ending read forming a chunk with up to max_hits
    //

    pipeline->chunk.read_begin = read_begin;

    // determine the index of the first hit in the chunk
    pipeline->chunk.mem_begin = mem->mem_filter.first_hit( read_begin );

    // make an iterator to count the number of hits in a given range of reads
    typedef thrust::counting_iterator<uint32>                            index_iterator;
    typedef thrust::transform_iterator<hit_count_functor,index_iterator> hit_counting_iterator;

    const hit_counting_iterator hit_counter(
        thrust::make_counting_iterator( 0u ),
        hit_count_functor( mem ) );

    // perform the binary search
    pipeline->chunk.read_end = uint32( nvbio::upper_bound(
            max_hits,
            hit_counter + read_begin,
            batch->size() - read_begin ) - hit_counter );

    // determine the index of the ending hit in the chunk
    pipeline->chunk.mem_end = mem->mem_filter.first_hit( pipeline->chunk.read_end );
}

// a functor to extract the reference location from a mem
struct mem_loc_functor
{
    typedef mem_state::mem_type argument_type;
    typedef uint64              result_type;

    NVBIO_HOST_DEVICE
    uint64 operator() (const argument_type mem) const
    {
        return uint64( mem.index_pos() ) | (uint64( mem.string_id() ) << 32);
    }
};

// locate all mems in the range defined by pipeline::chunk
void mem_locate(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;

    if (mem->mems.size() < command_line_options.mems_batch)
    {
        mem->mems.resize( command_line_options.mems_batch );
        mem->mems_index.resize( command_line_options.mems_batch );
        mem->mems_chain.resize( command_line_options.mems_batch );
    }

    const uint32 n_mems = pipeline->chunk.mem_end - pipeline->chunk.mem_begin;

    mem->mem_filter.locate(
        pipeline->chunk.mem_begin,
        pipeline->chunk.mem_end,
        mem->mems.begin() );

    // sort the mems by reference location
    nvbio::vector<device_tag,uint64> loc( n_mems );

    thrust::transform(
        mem->mems.begin(),
        mem->mems.begin() + n_mems,
        loc.begin(),
        mem_loc_functor() );

    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_mems,
        mem->mems_index.begin() );

    // TODO: this is slow, switch to nvbio::cuda::SortEnactor
    thrust::sort_by_key(
        loc.begin(),
        loc.begin() + n_mems,
        mem->mems_index.begin() );
}
