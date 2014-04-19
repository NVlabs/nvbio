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

#include "build-chains.h"
#include "mem-search.h"
#include "options.h"
#include "pipeline.h"

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/priority_queue.h>
#include <nvbio/basic/transform_iterator.h>
#include <nvbio/basic/vector_wrapper.h>
#include <nvbio/basic/cuda/primitives.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

using namespace nvbio;

// a functor to extract the read id from a mem
struct mem_read_id_functor
{
    typedef mem_state::mem_type argument_type;
    typedef uint32              result_type;

    NVBIO_HOST_DEVICE
    uint32 operator() (const argument_type mem) const { return mem.string_id(); }
};

// a class to keep track of a chain
struct chain
{
    // construct an empty chain
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    chain() : id(uint32(-1)) {}

    // construct a new chain from a single seed
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    chain(const uint32 _id, const mem_state::mem_type seed) :
        id( _id ),
        ref( seed.index_pos() ),
        span_beg( seed.span().x ),
        last_span( seed.span() )
    {}

    // test whether we can merge the given mem into this chain
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool merge(const mem_state::mem_type seed, const uint32 w, const uint32 max_chain_gap)
    {
        const uint32 seed_len = seed.span().y - seed.span().x;
        const uint32 last_len = last_span.y - last_span.x;
        const uint32 rbeg     = ref;
        const uint32 rend     = last_ref + last_len;

        // check whether seed is contained in the chain
        if (seed.span().x >= span_beg && seed.span().y <= last_span.y && seed.index_pos() >= rbeg && seed.index_pos() + seed_len <= rend)
            return true; // contained seed; do nothing

    	const int32 x = seed.span().x - last_span.x; // always non-negative
        const int32 y = seed.index_pos() - last_ref;
        if ((y >= 0) && (x - y <= w) && (x - last_len < max_chain_gap) && (y - last_len < max_chain_gap))
        {
             // grow the chain
            last_span = seed.span();
            last_ref  = seed.index_pos();
            return true;
        }
        return false;
    }

    uint32 id;              // chain id
    uint32 ref;             // reference coordinate of the first seed in the chain
    uint32 span_beg;        // read span begin
    uint32 last_ref;        // the reference coordinate of the last seed in the chain
    uint2  last_span;       // the read span of the last seed in the chain
};

struct chain_compare
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool operator() (const chain& chain1, const chain& chain2) const
    {
        // compare by the reference coordinate of the first seed of each chain
        return chain1.ref < chain2.ref;
    }
};

// build chains for the current pipeline::chunk of reads
__global__
void build_chains_kernel(
    const io::ReadDataDevice::const_plain_view_type batch,
    const read_chunk                                chunk,
    const uint32                                    pass_number,
    const uint32                                    n_active,
    const uint32*                                   active_reads,
          uint32*                                   active_flags,
    const uint32                                    w,
    const uint32                                    max_chain_gap,
    const uint32                                    n_mems,
    const mem_state::mem_type*                      mems,
    const uint32*                                   mems_index,
          uint64*                                   mems_chains)
{
    const uint32 thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= n_active)
        return;

    const uint32 read_id = active_reads[ thread_id ];

    // find the first seed belonging to this read
    const uint32 mem_begin = uint32( nvbio::lower_bound(
        read_id,
        nvbio::make_transform_iterator( mems, mem_read_id_functor() ),
        n_mems ) - nvbio::make_transform_iterator( mems, mem_read_id_functor() ) );

    // find the first seed belonging to the next read
    const uint32 mem_end = uint32( nvbio::lower_bound(
        read_id+1u,
        nvbio::make_transform_iterator( mems, mem_read_id_functor() ),
        n_mems ) - nvbio::make_transform_iterator( mems, mem_read_id_functor() ) );

    typedef nvbio::vector_wrapper<chain*> chain_vector_type;
    const uint32 MAX_CHAINS = 128;     // need to handle overflow in multiple passes...

    // keep a priority queue of the chains organized by the reference coordinate of their leftmost seed
    chain chain_queue_storage[MAX_CHAINS+1];
    nvbio::priority_queue<chain, chain_vector_type, chain_compare> chain_queue( chain_vector_type( 0u, chain_queue_storage ) );

    // keep track of the number of created chains
    uint64 n_chains = pass_number * MAX_CHAINS;

    // process the seeds in order
    for (uint32 i = nvbio::min( mem_begin, pass_number * MAX_CHAINS ); i < mem_end; ++i)
    {
        const uint32 seed_idx           = mems_index[i];
        const mem_state::mem_type seed  = mems[ seed_idx ];

        // the chain id for this seed, to be determined
        uint32 chain_id;

        // insert seed
        if (chain_queue.empty())
        {
            // get a new chain id
            chain_id = n_chains++;

            // build a new chain
            chain_queue.push( chain( chain_id, seed ) );
        }
        else
        {
            // find the closest chain...
            chain& chn = chain_queue.top();

            // and test whether we can merge this seed into it
            if (chn.merge( seed, w, max_chain_gap ) == false)
            {
                // get a new chain id
                chain_id = n_chains++;

                // build a new chain
                chain_queue.push( chain( chain_id, seed ) );
            }
            else
            {
                // merge with the existing chain
                chain_id = chn.id;
            }
        }

        // write out the chain id (OR'd with the read id)
        mems_chains[i] = n_chains | (uint64( read_id ) << 32);
    }

    // write out whether we need more passes
    active_flags[ thread_id ] =  pass_number * MAX_CHAINS < mem_end;
}

// build chains for the current pipeline::chunk of reads
void build_chains(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;

    const uint32 n_reads = pipeline->chunk.read_end - pipeline->chunk.read_begin;
    const uint32 n_mems  = pipeline->chunk.mem_end  - pipeline->chunk.mem_begin;

    const uint32 block_dim = 128;
    const uint32 n_blocks  = util::divide_ri( n_reads, block_dim );

    //
    // Here we are going to run multiple passes of the same kernel, as we cannot fit
    // all chains in local memory at once...
    //

    // prepare some ping-pong queues for tracking active reads that need more passes
    nvbio::vector<device_tag,uint32> active_reads( n_reads );
    nvbio::vector<device_tag,uint32> active_flags( n_reads );
    nvbio::vector<device_tag,uint32> out_reads( n_reads );
    nvbio::vector<device_tag,uint8>  temp_storage;

    // initialize the active reads queue
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + pipeline->chunk.read_begin,
        active_reads.begin() );

    uint32 n_active = n_reads;

    for (uint32 pass_number = 0u; n_active; ++pass_number)
    {
        // assign a chain id to each mem
        build_chains_kernel<<<n_blocks, block_dim>>>(
            nvbio::plain_view( *batch ),
            pipeline->chunk,
            pass_number,
            n_active,
            nvbio::plain_view( active_reads ),
            nvbio::plain_view( active_flags ),
            command_line_options.w,
            command_line_options.max_chain_gap,
            n_mems,
            nvbio::plain_view( mem->mems ),
            nvbio::plain_view( mem->mems_index ),
            nvbio::plain_view( mem->mems_chain ) );

        // shrink the set of active reads
        n_active = cuda::copy_flagged(
            n_active,
            active_reads.begin(),
            active_flags.begin(),
            out_reads.begin(),
            temp_storage );

        active_reads.swap( out_reads );
    }

    // sort mems by chain id
    thrust::sort_by_key( // TODO: this is slow, switch to nvbio::cuda::SortEnactor
        mem->mems_chain.begin(),
        mem->mems_chain.begin() + n_mems,
        mem->mems_index.begin() );
}
