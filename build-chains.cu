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

#include "build-chains.h"
#include "mem-search.h"
#include "options.h"
#include "pipeline.h"
#include "util.h"

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/priority_queue.h>
#include <nvbio/basic/timer.h>
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
    const io::ReadDataDevice::const_plain_view_type batch,              // the input batch of reads
    const read_chunk                                chunk,              // the current sub-batch
    const uint32                                    pass_number,        // the pass number - we process up to N seeds per pass
    const uint32                                    n_active,           // the number of active reads in this pass
    const uint32*                                   active_reads,       // the set of active reads
          uint8*                                    active_flags,       // the output set of active read flags
    const uint32                                    w,                  // w parameter
    const uint32                                    max_chain_gap,      // max chain gap parameter
    const uint32                                    n_mems,             // the total number of MEMs for this chunk of reads
    const mem_state::mem_type*                      mems,               // the MEMs for this chunk of reads
    const uint32*                                   mems_index,         // a sorting index into the MEMs specifying the processing order
          uint64*                                   mems_chains)        // the output chain IDs corresponding to the sorted MEMs
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

    // the maximum amount of chains we can output in one pass
    const uint32 MAX_CHAINS = 128;

    // keep a priority queue of the chains organized by the reference coordinate of their leftmost seed
    typedef nvbio::vector_wrapper<chain*>                                       chain_vector_type;
    typedef nvbio::priority_queue<chain, chain_vector_type, chain_compare>      chain_queue_type;

    chain            chain_queue_storage[MAX_CHAINS+1];
    chain_queue_type chain_queue( chain_vector_type( 0u, chain_queue_storage ) );

    // keep a counter tracking the number of chains that get created
    //
    // NOTE: here we conservatively assume that in the previous passes we have
    // created the maximum number of chains, so as to avoid assigning an already
    // taken ID to a new chain (which would result in merging potentially unrelated
    // chains)
    uint64 n_chains = pass_number * MAX_CHAINS;

    // compute the first and ending MEM to process in this pass
    const uint32 mem_batch_begin = mem_begin + pass_number * MAX_CHAINS;
    const uint32 mem_batch_end   = nvbio::min( mem_batch_begin + MAX_CHAINS, mem_end );

    // process the seeds in order
    for (uint32 i = mem_batch_begin; i < mem_batch_end; ++i)
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
            chain_queue_type::iterator chain_it = chain_queue.upper_bound( chain( 0u, seed ) );

            // and test whether we can merge this seed into it
            if (chain_it != chain_queue.end() &&
                chain_it->merge( seed, w, max_chain_gap ) == false)
            {
                // get a new chain id
                chain_id = n_chains++;

                // build a new chain
                chain_queue.push( chain( chain_id, seed ) );
            }
            else
            {
                // merge with the existing chain
                chain_id = chain_it->id;
            }
        }

        // write out the chain id (OR'd with the read id)
        mems_chains[i] = n_chains | (uint64( read_id ) << 32);
    }

    // write out whether we need more passes
    active_flags[ thread_id ] = (mem_batch_begin < mem_end) ? 1u : 0u;
}

// compute the coverage for each chain in a set
__global__
void chain_coverage_kernel(
    const uint32                                    n_chains,           // the number of chains
    const uint32*                                   chain_offsets,      // the chain offsets
    const uint32*                                   chain_lengths,      // the chain lengths
    const mem_state::mem_type*                      mems,               // the MEMs for this chunk of reads
    const uint32*                                   mems_index,         // a sorting index into the MEMs specifying their processing order
          uint2*                                    chain_ranges,       // the output chain ranges
          uint32*                                   chain_weights)      // the output chain weights
{
    const uint32 chain_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (chain_id >= n_chains)
        return;

    const uint32 begin = chain_offsets[ chain_id ];
    const uint32 end   = chain_lengths[ chain_id ] + begin;

    uint2  range  = make_uint2( uint32(-1), 0u );
    uint32 weight = 0;

    // NOTE: we assume here the MEMs of a chain appear sorted by their left coordinate
    for (uint32 i = begin; i < end; ++i)
    {
        const mem_state::mem_type seed = mems[ mems_index[i] ];

        const uint2 span = seed.span();

        if (span.x >= range.y)
            weight += span.y - span.x;
        else if (span.y > range.y)
            weight += span.y - range.y;

        range.x = nvbio::min( range.x, seed.span().x );
        range.y = nvbio::max( range.y, seed.span().y );
    }

    // write out the outputs
    chain_ranges[ chain_id ]  = range;
    chain_weights[ chain_id ] = weight;
}

// filter the chains belonging to each read
__global__
void chain_filter_kernel(
    const read_chunk                                chunk,              // the current sub-batch
    const uint32                                    n_chains,           // the number of chains
    const uint32*                                   chain_reads,        // the chain reads
    const uint32*                                   chain_offsets,      // the chain offsets
    const uint32*                                   chain_lengths,      // the chain lengths
    const uint32*                                   chain_index,        // the chain order
    const uint2*                                    chain_ranges,       // the chain ranges
    const uint32*                                   chain_weights,      // the chain weights
    const float                                     mask_level,         // input option
    const float                                     chain_drop_ratio,   // input option
    const uint32                                    min_seed_len,       // input option
          uint8*                                    chain_flags)        // the output flags
{
    const uint32 read_id = threadIdx.x + blockIdx.x * blockDim.x + chunk.read_begin;
    if (read_id >= chunk.read_end)
        return;

    const uint32 begin = uint32( nvbio::lower_bound( read_id, chain_reads, n_chains ) - chain_reads );
    const uint32 end   = uint32( nvbio::upper_bound( read_id, chain_reads, n_chains ) - chain_reads );

    // keep the first chain
    chain_flags[ chain_index[begin] ] = 1u; // mark to keep

    // and loop through all the rest to decide which ones to keep
    uint32 n = 1;

    for (uint32 i = begin + 1; i < end; ++i)
    {
        const uint2  i_span = chain_ranges[ chain_index[i] ];
        const uint32 i_w    = chain_weights[ chain_index[i] ];

        uint32 j;
        for (j = begin; j < begin + n; ++j)
        {
            const uint2  j_span = chain_ranges[ chain_index[j] ];
            const uint32 j_w    = chain_weights[ chain_index[j] ];

            const uint32 max_begin = nvbio::max( i_span.x, j_span.x );
            const uint32 min_end   = nvbio::min( i_span.y, j_span.y );

            if (min_end > max_begin) // have overlap
            {
                const uint32 min_l = nvbio::min( i_span.y - i_span.x, j_span.y - j_span.x );
				if (min_end - max_begin >= min_l * mask_level) // significant overlap
                {
                    chain_flags[ chain_index[i] ] = 1u; // mark to keep

                    if (i_w < j_w * chain_drop_ratio &&
                        j_w - i_w >= min_seed_len * 2)
                        break;
				}
            }
        }
		if (j == n) // no significant overlap with better chains, keep it.
        {
            chain_flags[ chain_index[i] ] = 1u; // mark to keep

            ++n;
        }
    }
}

// build chains for the current pipeline::chunk of reads
void build_chains(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    const ScopedTimer<float> timer( &pipeline->stats.chain_time ); // keep track of the time spent here

    struct mem_state *mem = &pipeline->mem;

    const uint32 n_reads = pipeline->chunk.read_end - pipeline->chunk.read_begin;
    const uint32 n_mems  = pipeline->chunk.mem_end  - pipeline->chunk.mem_begin;

    // skip pathological cases
    if (n_mems == 0u)
        return;

    //
    // Here we are going to run multiple passes of the same kernel, as we cannot fit
    // all chains in local memory at once...
    //

    // prepare some ping-pong queues for tracking active reads that need more passes
    nvbio::vector<device_tag,uint32> active_reads( n_reads );
    nvbio::vector<device_tag,uint8>  active_flags( n_reads );
    nvbio::vector<device_tag,uint32> out_reads( n_reads );
    nvbio::vector<device_tag,uint8>  temp_storage;

    // initialize the active reads queue
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u) + pipeline->chunk.read_begin,
        thrust::make_counting_iterator<uint32>(0u) + pipeline->chunk.read_end,
        active_reads.begin() );

    uint32 n_active = n_reads;

    for (uint32 pass_number = 0u; n_active; ++pass_number)
    {
        const uint32 block_dim = 128;
        const uint32 n_blocks  = util::divide_ri( n_active, block_dim );

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

        optional_device_synchronize();
        cuda::check_error("build-chains kernel");

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
    // NOTE: it's important here to use a stable-sort, so as to guarantee preserving
    // the ordering by left-coordinate of the MEMs
    thrust::sort_by_key(                                // TODO: this is slow, switch to nvbio::cuda::SortEnactor
        mem->mems_chain.begin(),
        mem->mems_chain.begin() + n_mems,
        mem->mems_index.begin() );

    optional_device_synchronize();
    nvbio::cuda::check_error("build-chains kernel");

    // extract the list of unique chain ids together with their counts, i.e. the chain lengths
    nvbio::vector<device_tag,uint64> unique_chains( n_mems );
    nvbio::vector<device_tag,uint32> unique_counts( n_mems );

    const uint32 n_chains = cuda::runlength_encode(
        n_mems,
        mem->mems_chain.begin(),
        unique_chains.begin(),
        unique_counts.begin(),
        temp_storage );

    // resize the chain vectors if needed
    uint32 reserved_space = uint32( mem->chain_lengths.size() );
    if (n_chains > reserved_space)
    {
        mem->chain_lengths.clear();
        mem->chain_lengths.resize( n_chains );
        mem->chain_offsets.clear();
        mem->chain_offsets.resize( n_chains );
        mem->chain_reads.clear();
        mem->chain_reads.resize( n_chains );

        reserved_space = n_chains;
    }

    // copy their lengths
    thrust::copy(
        unique_counts.begin(),
        unique_counts.begin() + n_chains,
        mem->chain_lengths.begin() );

    // find the offset to the beginning of each chain
    thrust::lower_bound(
        mem->mems_chain.begin(),
        mem->mems_chain.begin() + n_mems,
        unique_chains.begin(),
        unique_chains.begin() + n_chains,
        mem->chain_offsets.begin() );

    // extract the read-id frome the chain ids
    thrust::transform(
        unique_chains.begin(),
        unique_chains.begin() + n_chains,
        mem->chain_reads.begin(),
        nvbio::hi_bits_functor<uint32,uint64>() );

    nvbio::vector<device_tag,uint2>  chain_ranges( n_chains );
    nvbio::vector<device_tag,uint32> chain_weights( n_chains );
    nvbio::vector<device_tag,uint32> chain_index( reserved_space ); // potentially a little bigger because we'll reuse
                                                                    // it for the final filtering...

    optional_device_synchronize();
    cuda::check_error("chain-coverage-init");

    // compute chain coverages
    {
        const uint32 block_dim = 128;
        const uint32 n_blocks  = util::divide_ri( n_chains, block_dim );

        chain_coverage_kernel<<<n_blocks, block_dim>>>(
            n_chains,
            nvbio::plain_view( mem->chain_offsets ),
            nvbio::plain_view( mem->chain_lengths ),
            nvbio::plain_view( mem->mems ),
            nvbio::plain_view( mem->mems_index ),
            nvbio::plain_view( chain_ranges ),
            nvbio::plain_view( chain_weights ) );

        optional_device_synchronize();
        cuda::check_error("chain-coverage kernel");
    }

    // sort the chains by weight
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_chains,
        chain_index.begin() );

    thrust::sort_by_key(                            // TODO: this is slow, switch to nvbio::cuda::SortEnactor
        chain_weights.begin(),
        chain_weights.begin() + n_chains,
        chain_index.begin() );

    nvbio::vector<device_tag,uint8> chain_flags( n_chains );
    thrust::fill( chain_flags.begin(), chain_flags.begin() + n_chains, 0u );

    // filter chains: set the flags for the chains to be kept
    {
        const uint32 block_dim = 128;
        const uint32 n_blocks  = util::divide_ri( n_reads, block_dim );

        chain_filter_kernel<<<n_blocks, block_dim>>>(
            pipeline->chunk,
            n_chains,
            nvbio::plain_view( mem->chain_reads ),
            nvbio::plain_view( mem->chain_offsets ),
            nvbio::plain_view( mem->chain_lengths ),
            nvbio::plain_view( chain_index ),
            nvbio::plain_view( chain_ranges ),
            nvbio::plain_view( chain_weights ),
            command_line_options.mask_level,
            command_line_options.chain_drop_ratio,
            command_line_options.min_seed_len,
            nvbio::plain_view( chain_flags ) );

        optional_device_synchronize();
        cuda::check_error("chain-dilter kernel");
    }

    // filter chain_reads
    const uint32 n_filtered_chains = cuda::copy_flagged(
        n_chains,
        mem->chain_reads.begin(),
        chain_flags.begin(),
        chain_index.begin(),
        temp_storage );

    mem->chain_reads.swap( chain_index );

    // filter chain_offsets
    cuda::copy_flagged(
        n_chains,
        mem->chain_offsets.begin(),
        chain_flags.begin(),
        chain_index.begin(),
        temp_storage );

    mem->chain_offsets.swap( chain_index );

    // filter chain_lengths
    cuda::copy_flagged(
        n_chains,
        mem->chain_lengths.begin(),
        chain_flags.begin(),
        chain_index.begin(),
        temp_storage );

    mem->chain_lengths.swap( chain_index );

    log_verbose(stderr, "  chains: %u -> %u\n", n_chains, n_filtered_chains);
}
