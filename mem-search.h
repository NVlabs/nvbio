/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <nvbio/io/fmi.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/fmindex/mem.h>

using namespace nvbio;

struct chain_coverage
{
    uint32 begin;
    uint32 end;
    uint32 weight;
    uint32 overlap;
};

struct mem_state
{
    typedef io::FMIndexDataDevice::fm_index_type                 fm_index_type;
    typedef MEMFilterDevice<fm_index_type>                       mem_filter_type;
    typedef nvbio::vector<device_tag, mem_filter_type::mem_type> mem_vector_type;
    typedef io::FMIndexDataDevice::stream_type                   genome_type;
    typedef mem_filter_type::mem_type                            mem_type;

    nvbio::io::FMIndexData       *fmindex_data_host;
    nvbio::io::FMIndexDataDevice *fmindex_data_device;

    // the FM-index objects
    fm_index_type f_index, r_index;

    // our MEM filter object, used to rank and locate MEMs and keep track of statistics
    mem_filter_type mem_filter;

    // the result vector for mem_search
    mem_vector_type mems;

    // a sorting index into the mems (first by reference location, then by chain id)
    nvbio::vector<device_tag,uint32> mems_index;

    // the chain IDs of each mem
    nvbio::vector<device_tag,uint64> mems_chain;

    // the list of chains
    nvbio::vector<device_tag,uint32> chain_offsets;     // the first seed of each chain
    nvbio::vector<device_tag,uint32> chain_lengths;     // the number of seeds in each chain
    nvbio::vector<device_tag,uint32> chain_reads;       // the read (strand) id of each chain
};

struct read_chunk
{
    read_chunk() :
        read_begin(0),
        read_end(0),
        mem_begin(0),
        mem_end(0) {}

    // ID of the first and last reads in this chunk
    uint32  read_begin;
    uint32  read_end;

    // index of the first hit for the first read
    uint32  mem_begin;
    // index of the last hit for the last read
    uint32  mem_end;
};

void mem_init(struct pipeline_context *pipeline);
void mem_search(struct pipeline_context *pipeline, const io::ReadDataDevice *batch);

// given the first read in a chunk, determine a suitably sized chunk of reads
// (for which we can locate all MEMs in one go), updating pipeline::chunk
void fit_read_chunk(
    struct pipeline_context     *pipeline,
    const io::ReadDataDevice    *batch,
    const uint32                read_begin);    // first read in the chunk

// locate all mems in the range defined by pipeline::chunk
void mem_locate(struct pipeline_context *pipeline, const io::ReadDataDevice *batch);
