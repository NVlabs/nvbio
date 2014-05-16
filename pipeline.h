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

#include <nvbio/io/output/output_file.h>
#include <nvbio/io/fmi.h>
#include <nvbio/fmindex/mem.h>

#include "mem-search.h"

using namespace nvbio;

/// pipeline stats
///
struct pipeline_stats
{
    pipeline_stats() :
        time        ( 0.0f ),
        io_time     ( 0.0f ),
        search_time ( 0.0f ),
        locate_time ( 0.0f ),
        chain_time  ( 0.0f ),
        n_reads     ( 0 ),
        n_mems      ( 0 ),
        n_chains    ( 0 )
    {}

    float time;
    float io_time;
    float search_time;
    float locate_time;
    float chain_time;

    uint64 n_reads;
    uint64 n_mems;
    uint64 n_chains;
};

struct mem_state
{
    typedef nvbio::io::FMIndexDataDevice::fm_index_type          fm_index_type;
    typedef nvbio::MEMFilterDevice<fm_index_type>                mem_filter_type;
    typedef nvbio::io::FMIndexDataDevice::stream_type            genome_type;
    typedef mem_filter_type::mem_type                            mem_type;

    nvbio::io::FMIndexData       *fmindex_data_host;
    nvbio::io::FMIndexDataDevice *fmindex_data_device;

    fm_index_type                    f_index;           ///< the forward FM-index object
    fm_index_type                    r_index;           ///< the reverse FM-index object

    mem_filter_type                  mem_filter;        ///< our MEM filter object, used to rank and locate MEMs and keep track of statistics
};

/// the state of the MEM chains relative to a set of reads
///
struct chains_state
{
    typedef mem_state::mem_type                     mem_type;
    typedef nvbio::vector<device_tag, mem_type>     mem_vector_type;

    mem_vector_type                  mems;              ///< the result vector for mem_search

    nvbio::vector<device_tag,uint32> mems_index;        ///< a sorting index into the mems (initially by reference location, then by chain id)
    nvbio::vector<device_tag,uint64> mems_chain;        ///< the chain IDs of each mem

    // the list of chains
    nvbio::vector<device_tag,uint32> chain_offsets;     ///< the first seed of each chain
    nvbio::vector<device_tag,uint32> chain_lengths;     ///< the number of seeds in each chain
    nvbio::vector<device_tag,uint32> chain_reads;       ///< the read (strand) id of each chain
    uint32                           n_chains;          ///< the number of chains
};

/// a small object acting as a "reference" for a chain
///
struct chain_reference;

/// a POD structure to view chains as if they were stored in AOS format;
/// unlike chains_state, this class can be passed as a kernel parameter and its members
/// can be accessed from the device.
///
struct chains_view
{
    typedef chains_state::mem_type                                  mem_type;
    //typedef nvbio::vector<device_tag,uint32>::const_plain_view_type index_vector_type;
    //typedef chains_state::mem_vector_type::const_plain_view_type    mem_vector_type;
    typedef const uint32*                                           index_vector_type;
    typedef const mem_type*                                         mem_vector_type;

    /// constructor
    ///
    chains_view(const chains_state& state) :
        mems( plain_view( state.mems ) ),
        mems_index( plain_view( state.mems_index ) ),
        chain_offsets( plain_view( state.chain_offsets ) ),
        chain_lengths( plain_view( state.chain_lengths ) ),
        chain_reads( plain_view( state.chain_reads ) ),
        n_chains( state.n_chains )
    {}

    /// return a "reference" to the i-th chain
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    chain_reference operator[] (const uint32 i) const;

    mem_vector_type   mems;                         ///< the result vector for mem_search
    index_vector_type mems_index;                   ///< a sorting index into the mems (initially by reference location, then by chain id)

    index_vector_type chain_offsets;                ///< the first seed of each chain
    index_vector_type chain_lengths;                ///< the number of seeds in each chain
    index_vector_type chain_reads;                  ///< the read (strand) id of each chain
    uint32            n_chains;                     ///< the number of chains
};

/// a small object acting as a "reference" for a chain, allowing to view it as if was a single object
///
struct chain_reference
{
    typedef chains_state::mem_type mem_type;

    /// constructor
    ///
    NVBIO_HOST_DEVICE
    chain_reference(const chains_view& _chains, const uint32 _i) : chains(_chains), idx(_i) {}

    /// return the read this chain belongs to
    ///
    NVBIO_HOST_DEVICE
    uint32 read() const { return chains.chain_reads[ idx ]; }

    /// return the number of seeds in this chain
    ///
    NVBIO_HOST_DEVICE
    uint32 size() const { return chains.chain_lengths[ idx ]; }

    /// return the i-th seed in this chain
    ///
    NVBIO_HOST_DEVICE
    mem_type operator[] (const uint32 i) const
    {
        // grab the offset to the first seed of this chain
        const uint32 offset = chains.chain_offsets[ idx ];

        // return the requested seed, remembering they are sorted by chain through an index
        return chains.mems[ chains.mems_index[ offset + i ] ];
    }

    const chains_view& chains;
    const uint32       idx;
};

// return a "reference" to the i-th chain
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
chain_reference chains_view::operator[] (const uint32 i) const
{
    return chain_reference( *this, i );
}

/// a contiguous subset of reads from a batch associated with all their MEMs
///
struct read_chunk
{
    read_chunk() :
        read_begin(0),
        read_end(0),
        mem_begin(0),
        mem_end(0) {}

    uint32  read_begin;         ///< ID of the first read in this chunk
    uint32  read_end;           ///< ID of the ending read in this chunk

    uint32  mem_begin;          ///< index of the first hit for the first read
    uint32  mem_end;            ///< index of the last hit for the last read
};

/// the alignment sub-pipeline state
///
/// during alignment, we essentially keep a queue of "active" reads, corresponding
/// to those reads for which there's more chains to process; at every step, we select
/// one new chain from each read as an alignment candidate, removing it from the set.
/// This is done keeping a set of (begin,end) pointers per read and advancing the
/// begin field - when a range becomes empty, it's removed
///
struct alignment_state
{
    uint32                               n_active;              ///< the number of active reads in the alignment queue
    nvbio::vector<device_tag,uint32>     begin_chains;          ///< the first chain for each read in the processing queue
    nvbio::vector<device_tag,uint32>     end_chains;            ///< the ending chain for each read in the processing queue
    nvbio::vector<device_tag,uint2>      query_spans;           ///< the query chain spans
    nvbio::vector<device_tag,uint2>      ref_spans;             ///< the reference chain spans
    nvbio::vector<device_tag,uint32>     temp_queue;            ///< a temporary queue
    nvbio::vector<device_tag,uint32>     stencil;               ///< a temporary stencil vector
};

/// the state of the pipeline
///
struct pipeline_state 
{
    nvbio::io::OutputFile*  output;
    mem_state               mem;
    chains_state            chn;
    alignment_state         aln;
    read_chunk              chunk;
    pipeline_stats          stats;
};
