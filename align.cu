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

#include "align.h"
#include "mem-search.h"
#include "options.h"
#include "pipeline.h"
#include "util.h"

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/transform_iterator.h>
#include <nvbio/basic/vector_view.h>
#include <nvbio/basic/primitives.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

using namespace nvbio;

// initialize the alignment pipeline
//
void align_init(struct pipeline_state *pipeline, const io::SequenceDataDevice<DNA_N> *batch)
{
    struct chains_state<device_tag>    *chn = &pipeline->chn;
    struct alignment_state<device_tag> *aln = &pipeline->aln;

    const uint32 n_reads  = pipeline->chunk.read_end - pipeline->chunk.read_begin;
    const uint32 n_chains = chn->n_chains;

    // initially, target the device
    pipeline->system = DEVICE;

    // reserve enough storage
    if (aln->stencil.size() < n_reads)
    {
        aln->begin_chains.clear(); aln->begin_chains.resize( n_reads );
        aln->end_chains.clear();   aln->end_chains.resize( n_reads );
        aln->stencil.clear();      aln->stencil.resize( n_reads );
        aln->temp_queue.clear();   aln->temp_queue.resize( n_reads );
        aln->query_spans.clear();  aln->query_spans.resize( n_reads );
        aln->ref_spans.clear();    aln->ref_spans.resize( n_reads );
    }

    // find the first chain for each read
    thrust::lower_bound(
        chn->chain_reads.begin(),
        chn->chain_reads.begin() + n_chains,
        thrust::make_counting_iterator<uint32>( pipeline->chunk.read_begin ),
        thrust::make_counting_iterator<uint32>( pipeline->chunk.read_end ),
        aln->begin_chains.begin() );

    // find the ending chain for each read
    thrust::upper_bound(
        chn->chain_reads.begin(),
        chn->chain_reads.begin() + n_chains,
        thrust::make_counting_iterator<uint32>( pipeline->chunk.read_begin ),
        thrust::make_counting_iterator<uint32>( pipeline->chunk.read_end ),
        aln->end_chains.begin() );

    aln->n_active = n_reads;
}

// a functor to extract the query span from a chain
//
struct query_span_functor
{
    NVBIO_HOST_DEVICE
    query_span_functor(const chains_view _chains) : chains( _chains ) {}

    // the functor operator
    NVBIO_HOST_DEVICE
    uint2 operator() (const uint32 idx) const
    {
        const chain_reference chain = chains[idx];

        const uint32 len = chain.size();

        uint2 span = make_uint2( uint32(-1), 0u );

        // loop through all seeds in this chain
        for (uint32 i = 0; i < len; ++i)
        {
            // fetch the i-th seed
            const chains_view::mem_type seed = chain[i];

            span.x = nvbio::min( span.x, seed.span().x );
            span.y = nvbio::max( span.y, seed.span().y );
        }
        return span;
    }

    const chains_view chains;
};

// a functor to extract the reference span from a chain
//
struct ref_span_functor
{
    NVBIO_HOST_DEVICE
    ref_span_functor(const chains_view _chains) : chains( _chains ) {}

    // the functor operator
    NVBIO_HOST_DEVICE
    uint2 operator() (const uint32 idx) const
    {
        const chain_reference chain = chains[idx];

        const uint32 len = chain.size();

        uint2 span = make_uint2( uint32(-1), 0u );

        // loop through all seeds in this chain
        for (uint32 i = 0; i < len; ++i)
        {
            // fetch the i-th seed
            const chains_view::mem_type seed = chain[i];

            span.x = nvbio::min( span.x, seed.index_pos() );
            span.y = nvbio::max( span.y, seed.index_pos() + seed.span().y - seed.span().x );
        }
        return span;
    }

    const chains_view chains;
};

// perform banded alignment
//
template <typename system_tag>
uint32 align_short(
    chains_state<system_tag>            *chn,
    alignment_state<system_tag>         *aln,
    const io::SequenceDataDevice<DNA_N> *batch)
{
    //
    // During alignment, we essentially keep a queue of "active" reads, corresponding
    // to those reads for which there's more chains to process; at every step, we select
    // one new chain from each read as an alignment candidate, removing it from the set.
    // This is done keeping a set of (begin,end) pointers per read and advancing the
    // begin field - when a range becomes empty, it's removed
    //
    uint32 n_active = aln->n_active;

    // build a stencil of the active reads, stencil[i] = (begin_chains[i] != end_chains[i])
    thrust::transform(
        aln->begin_chains.begin(),
        aln->begin_chains.begin() + n_active,
        aln->end_chains.begin(),
        aln->stencil.begin(),
        nvbio::not_equal_functor<uint32>() );

    nvbio::vector<system_tag,uint8> temp_storage;

    // filter away reads that are done processing because there's no more chains
    copy_flagged(
        n_active,
        aln->begin_chains.begin(),
        aln->stencil.begin(),
        aln->temp_queue.begin(),
        temp_storage );

    aln->begin_chains.swap( aln->temp_queue );

    n_active = copy_flagged(
        n_active,
        aln->end_chains.begin(),
        aln->stencil.begin(),
        aln->temp_queue.begin(),
        temp_storage );

    aln->end_chains.swap( aln->temp_queue );

    // reset the number of active reads
    aln->n_active = n_active;

    // check whether there's no more work to do
    if (n_active == 0)
        return 0u;

    // now build a view of the chains
    const chains_view chains( *chn );

    const nvbio::vector<system_tag,uint32>&     cur_chains  = aln->begin_chains;
          nvbio::vector<system_tag,uint2>&      query_spans = aln->query_spans;
          nvbio::vector<system_tag,uint2>&      ref_spans   = aln->ref_spans;

    // compute the chain query-spans
    thrust::transform(
        cur_chains.begin(),
        cur_chains.begin() + n_active,
        query_spans.begin(),
        query_span_functor( chains ) );

    // compute the chain reference-spans
    thrust::transform(
        cur_chains.begin(),
        cur_chains.begin() + n_active,
        ref_spans.begin(),
        ref_span_functor( chains ) );

    // add one to the processed chains
    thrust::transform(
        aln->begin_chains.begin(),
        aln->begin_chains.begin() + n_active,
        thrust::make_constant_iterator<uint32>( 1u ),
        aln->begin_chains.begin(),
        nvbio::add_functor() );

    return n_active;
}

// perform banded alignment
//
uint32 align_short(pipeline_state *pipeline, const io::SequenceDataDevice<DNA_N> *batch)
{
    if (pipeline->system == DEVICE &&       // if currently on the device,
        pipeline->aln.n_active < 16*1024)   // but too little parallelism...
    {
        // copy the state of the pipeline to the host
        pipeline->system = HOST;
        pipeline->h_chn = pipeline->chn;
        pipeline->h_aln = pipeline->aln;
    }

    if (pipeline->system == HOST)
        return align_short( &pipeline->h_chn, &pipeline->h_aln, batch );
    else
        return align_short( &pipeline->chn, &pipeline->aln, batch );
}
