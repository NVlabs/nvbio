/*
 * nvbio
 * Copyright (c) 2011-2014, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *    * Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *    * Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *    * Neither the name of the NVIDIA CORPORATION nor the
 *      names of its contributors may be used to endorse or promote products
 *      derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <nvbio/sufsort/sufsort_priv.h>
#include <nvbio/sufsort/compression_sort.h>
#include <nvbio/sufsort/prefix_doubling_sufsort.h>
#include <nvbio/basic/string_set.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/cuda/sort.h>
#include <nvbio/basic/timer.h>
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <mgpuhost.cuh>
#include <moderngpu.cuh>


namespace nvbio {

namespace cuda {

/// return the position of the primary suffix of a string
///
template <typename string_type>
typename string_type::index_type find_primary(
    const typename string_type::index_type  string_len,
    const string_type                       string)
{
    const uint32 SYMBOL_SIZE = string_type::SYMBOL_SIZE;

    // compute the primary by simply counting how many of the suffixes between 1 and N
    // are lexicographically less than the primary suffix
    return thrust::transform_reduce(
        thrust::make_counting_iterator<uint32>(1u),
        thrust::make_counting_iterator<uint32>(0u) + string_len,
        bind_second_functor< priv::string_suffix_less<SYMBOL_SIZE,string_type> >(
            priv::string_suffix_less<SYMBOL_SIZE,string_type>( string_len, string ),
            0u ),
        0u,
        thrust::plus<uint32>() ) + 1u;
}

/// Sort the suffixes of all the strings in the given string_set
///
template <typename string_set_type, typename output_handler>
void suffix_sort(
    const string_set_type&   string_set,
          output_handler&    output,
          BWTParams*         params)
{
    typedef uint32 word_type;
    const uint32 WORD_BITS   = uint32( 8u * sizeof(word_type) );
    const uint32 DOLLAR_BITS = 4;

    const uint32 SYMBOL_SIZE      = 2u;
    const uint32 SYMBOLS_PER_WORD = priv::symbols_per_word<SYMBOL_SIZE,WORD_BITS,DOLLAR_BITS>();

    const uint32 n = string_set.size();

    int current_device;
    cudaGetDevice( &current_device );
    mgpu::ContextPtr mgpu_ctxt = mgpu::CreateCudaDevice( current_device ); 

    // instantiate a suffix flattener on the string set
    priv::SetSuffixFlattener<SYMBOL_SIZE> suffixes( mgpu_ctxt );
    suffixes.set( string_set );

    // compute the maximum number of words needed to represent a suffix
    const uint32 m = (suffixes.max_length( string_set ) + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

    // compute the number of suffixes
    const uint32 n_suffixes = suffixes.n_suffixes;

    thrust::device_vector<word_type> radices( n_suffixes*2 );
    thrust::device_vector<uint32>    indices( n_suffixes*2 );

    // initialize the list of suffix indices
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(n_suffixes),
        indices.begin() );

    cuda::SortBuffers<word_type*,uint32*> sort_buffers;
    cuda::SortEnactor                     sort_enactor;

    sort_buffers.selector  = 0;
    sort_buffers.keys[0]   = nvbio::device_view( radices );
    sort_buffers.keys[1]   = nvbio::device_view( radices ) + n_suffixes;
    sort_buffers.values[0] = nvbio::device_view( indices );
    sort_buffers.values[1] = nvbio::device_view( indices ) + n_suffixes;

    // do what is essentially an LSD radix-sort on the suffixes, word by word
    for (int32 word_idx = m-1; word_idx >= 0; --word_idx)
    {
        // extract the given radix word from each of the partially sorted suffixes
        suffixes.flatten(
            string_set,
            word_idx,
            priv::Bits<WORD_BITS,DOLLAR_BITS>(),
            indices.begin() + sort_buffers.selector * n_suffixes,
            radices.begin() + sort_buffers.selector * n_suffixes );

        // and sort them
        sort_enactor.sort( n_suffixes, sort_buffers );
    }

    output.process(
        n_suffixes,
        nvbio::device_view( indices ) + sort_buffers.selector * n_suffixes,
        nvbio::device_view( suffixes.string_ids ),
        nvbio::device_view( suffixes.cum_lengths ));
}

/// Sort all the suffixes of a given string
///
template <typename string_type, typename output_iterator>
void suffix_sort(
    const typename stream_traits<string_type>::index_type   string_len,
    const string_type                                       string,
    output_iterator                                         output,
    BWTParams*                                              params)
{
    PrefixDoublingSufSort sufsort;
    sufsort.sort(
        string_len,
        string,
        output + 1u );

    // assign the zero'th suffix
    output[0] = string_len;

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    extract  : %5.1f ms\n", 1.0e3f * sufsort.extract_time) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    gather   : %5.1f ms\n", 1.0e3f * sufsort.gather_time) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    r-sort   : %5.1f ms\n", 1.0e3f * sufsort.radixsort_time) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    segment  : %5.1f ms\n", 1.0e3f * sufsort.segment_time) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    invert   : %5.1f ms\n", 1.0e3f * sufsort.inverse_time) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    compact  : %5.1f ms\n", 1.0e3f * sufsort.compact_time) );
}

/// Compute the bwt of a device-side string
///
/// \return         position of the primary suffix / $ symbol
///
template <typename string_type, typename output_iterator>
typename string_type::index_type bwt(
    const typename string_type::index_type  string_len,
    string_type                             string,
    output_iterator                         output,
    BWTParams*                              params)
{
    typedef typename string_type::index_type index_type;
    const uint32 SYMBOL_SIZE = string_type::SYMBOL_SIZE;

    const uint32     M = 256*1024;
    const index_type N = string_len;

    const uint32 n_chunks = (N + M-1) / M;

    const uint32 BUCKETING_BITS = 20;
    const uint32 DOLLAR_BITS    = 4;

    priv::StringSuffixBucketer<SYMBOL_SIZE,BUCKETING_BITS,DOLLAR_BITS> bucketer;

    typedef uint32 word_type;
    const uint32   WORD_BITS = uint32( 8u * sizeof(uint32) );

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    //const size_t max_super_block_mem  = free - max_block_size*16u - 512u*1024u*1024u;
    //const uint32 max_super_block_size = uint32( max_super_block_mem / 4u );
    const uint32 max_super_block_size = nvbio::min(             // requires max_super_block_size*4 host memory bytes
        index_type( params ?
            (params->host_memory - (128u*1024u*1024u)) / 4u :   // leave 128MB for the bucket counters
            512*1024*1024 ),
        string_len );

    const uint32 max_block_size = 32*1024*1024;                 // requires max_block_size*21 device memory bytes
    const uint32 DELAY_BUFFER   = 512*1024;

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  super-block-size: %.1f M\n", float(max_super_block_size)/float(1024*1024)) );
    thrust::host_vector<uint32> h_super_suffixes( max_super_block_size, 0u );
    thrust::host_vector<uint32> h_block_suffixes( max_block_size );
    thrust::host_vector<uint32> h_block_radices( max_block_size );

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  device-alloc(%.1f GB)... started\n", float(max_block_size*21u)/float(1024*1024*1024)) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    free: %.1f GB\n", float(free)/float(1024*1024*1024)) );

    thrust::device_vector<uint32> d_super_suffixes( max_block_size );
    thrust::device_vector<uint32> d_delayed_suffixes( max_block_size );
    thrust::device_vector<uint32> d_delayed_slots( max_block_size );
    thrust::device_vector<uint8>  d_block_bwt( max_block_size );

    int current_device;
    cudaGetDevice( &current_device );
    mgpu::ContextPtr mgpu_ctxt = mgpu::CreateCudaDevice( current_device ); 

  #define COMPRESSION_SORTING
  #if defined(COMPRESSION_SORTING)
    CompressionSort compression_sort( mgpu_ctxt );
    compression_sort.reserve( max_block_size );
  #endif

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  device-alloc(%.1f GB)... done\n", float(max_super_block_size*8u + max_block_size*16u)/float(1024*1024*1024)) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  bucket counting\n") );

    // global bucket sizes
    thrust::device_vector<uint32> d_buckets( 1u << (BUCKETING_BITS), 0u );

    for (uint32 chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
    {
        const index_type chunk_begin = chunk_id * M;
        const index_type chunk_end   = nvbio::min( chunk_begin + M, N );
        const uint32     chunk_size  = uint32( chunk_end - chunk_begin );

        // assemble the device chunk string
        const string_type d_chunk = string + chunk_begin;

        // count the chunk's buckets
        bucketer.count( chunk_size, N - chunk_begin, d_chunk );

        // and merge them in with the global buckets
        thrust::transform(
            bucketer.d_buckets.begin(),
            bucketer.d_buckets.end(),
            d_buckets.begin(),
            d_buckets.begin(),
            thrust::plus<uint32>() );
    }

    thrust::host_vector<uint32> h_buckets( d_buckets );
    thrust::host_vector<uint32> h_bucket_offsets( d_buckets.size() );
    thrust::host_vector<uint32> h_subbuckets( d_buckets.size() );

    const uint32 max_bucket_size = thrust::reduce(
        d_buckets.begin(),
        d_buckets.end(),
        0u,
        thrust::maximum<uint32>() );

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    max bucket size: %u\n", max_bucket_size) );
    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      c-sort  : %.1fs\n", bucketer.d_count_sort_time) );

    //
    // at this point, we have to do multiple passes through the input string set,
    // collecting in each pass as many buckets as we can fit in memory at once
    //

    // scan the bucket offsets so as to have global positions
    thrust::exclusive_scan(
        h_buckets.begin(),
        h_buckets.end(),
        h_bucket_offsets.begin() );

    // build the subbucket pointers
    for (uint32 bucket_begin = 0, bucket_end = 0; bucket_begin < h_buckets.size(); bucket_begin = bucket_end)
    {
        // grow the block of buckets until we can
        uint32 bucket_size;
        for (bucket_size = 0; (bucket_end < h_buckets.size()) && (bucket_size + h_buckets[bucket_end] < max_super_block_size); ++bucket_end)
            bucket_size += h_buckets[bucket_end];

        // build the sub-buckets
        for (uint32 subbucket_begin = bucket_begin, subbucket_end = bucket_begin; subbucket_begin < bucket_end; subbucket_begin = subbucket_end)
        {
            // grow the block of sub-buckets until we can
            uint32 subbucket_size;
            for (subbucket_size = 0; (subbucket_end < bucket_end) && (subbucket_size + h_buckets[subbucket_end] < max_block_size); ++subbucket_end)
            {
                subbucket_size += h_buckets[subbucket_end];

                h_subbuckets[ subbucket_end ] = subbucket_begin; // point to the beginning of this sub-bucket
            }
        }
    }

    // build the subbucket pointers
    thrust::device_vector<uint32> d_subbuckets( h_subbuckets );

    NVBIO_VAR_UNUSED float sufsort_time     = 0.0f;
    NVBIO_VAR_UNUSED float collect_time     = 0.0f;
    NVBIO_VAR_UNUSED float bwt_copy_time    = 0.0f;
    NVBIO_VAR_UNUSED float bwt_scatter_time = 0.0f;

    index_type global_suffix_offset = 0;

    const index_type NULL_PRIMARY = index_type(-1);
    index_type primary = NULL_PRIMARY;

    // encode the first BWT symbol explicitly
    priv::device_copy( 1u, string + N-1, output, index_type(0u) );

    for (uint32 bucket_begin = 0, bucket_end = 0; bucket_begin < h_buckets.size(); bucket_begin = bucket_end)
    {
        // grow the block of buckets until we can
        uint32 bucket_size;
        for (bucket_size = 0; (bucket_end < h_buckets.size()) && (bucket_size + h_buckets[bucket_end] < max_super_block_size); ++bucket_end)
            bucket_size += h_buckets[bucket_end];

        uint32 suffix_count = 0;

        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  collect buckets[%u:%u] (%u suffixes)\n", bucket_begin, bucket_end, bucket_size) );
        Timer collect_timer;
        collect_timer.start();

        for (uint32 chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
        {
            const index_type chunk_begin = chunk_id * M;
            const index_type chunk_end   = nvbio::min( chunk_begin + M, N );
            const uint32     chunk_size  = uint32( chunk_end - chunk_begin );

            // assemble the device chunk string
            const string_type d_chunk = string + chunk_begin;

            // collect the chunk's suffixes within the bucket range
            const uint32 n_collected = bucketer.collect(
                chunk_size,
                N - chunk_begin,
                d_chunk,
                bucket_begin,
                bucket_end,
                chunk_begin,
                d_subbuckets.begin(),
                h_block_radices.begin(),
                h_block_suffixes.begin() );

            // dispatch each suffix to their respective bucket
            for (uint32 i = 0; i < n_collected; ++i)
            {
                const uint32 bucket = h_block_radices[i];
                const uint32 slot   = h_bucket_offsets[bucket]++;
                h_super_suffixes[ slot - global_suffix_offset ] = h_block_suffixes[i];
            }

            suffix_count += n_collected;

            if (suffix_count > max_super_block_size)
            {
                log_error(stderr,"buffer size exceeded! (%u/%u)\n", suffix_count, max_block_size);
                exit(1);
            }
        }
        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        collect_timer.stop();
        collect_time += collect_timer.seconds();

        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  collect : %.1fs, %.1f M suffixes/s\n", collect_time, 1.0e-6f*float(global_suffix_offset + suffix_count)/collect_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    setup   : %.1fs\n", bucketer.d_setup_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    flatten : %.1fs\n", bucketer.d_flatten_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    b-sort  : %.1fs\n", bucketer.d_collect_sort_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    search  : %.1fs\n", bucketer.d_search_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    copy    : %.1fs\n", bucketer.d_copy_time) );

        //
        // at this point we have a large collection of localized suffixes to sort in d_super_suffixes;
        // we'll do it looping on multiple sub-buckets, on the GPU
        //

        const uint32 SYMBOLS_PER_WORD = priv::symbols_per_word<SYMBOL_SIZE, WORD_BITS,DOLLAR_BITS>();

        suffix_count = 0u;

        uint32 delay_count = 0;

        for (uint32 subbucket_begin = bucket_begin, subbucket_end = bucket_begin; subbucket_begin < bucket_end; subbucket_begin = subbucket_end)
        {
            // grow the block of sub-buckets until we can
            uint32 subbucket_size;
            for (subbucket_size = 0; (subbucket_end < bucket_end) && (subbucket_size + h_buckets[subbucket_end] < max_block_size); ++subbucket_end)
                subbucket_size += h_buckets[subbucket_end];

            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"\r  sufsort buckets[%u:%u] (%u suffixes, %.1f M suffixes/s)        ", subbucket_begin, subbucket_end, subbucket_size, 1.0e-6f*float(global_suffix_offset + suffix_count)/sufsort_time ) );

            // consume subbucket_size suffixes
            const uint32 n_suffixes = subbucket_size;

            Timer timer;
            timer.start();

            // initialize the device sorting indices
            thrust::copy(
                h_super_suffixes.begin() + suffix_count,
                h_super_suffixes.begin() + suffix_count + n_suffixes,
                d_super_suffixes.begin() );

        #if defined(COMPRESSION_SORTING)
            compression_sort.sort(
                string_len,
                string,
                n_suffixes,
                d_super_suffixes.begin(),
                delay_count,
                suffix_count,
                d_delayed_suffixes.begin(),
                d_delayed_slots.begin() );
        #else // if defined(COMPRESSION_SORTING)
            // and sort the corresponding suffixes
            thrust::stable_sort(
                d_super_suffixes.begin(),
                d_super_suffixes.begin() + n_suffixes,
                priv::string_suffix_less<SYMBOL_SIZE,string_type>( N, string ) );

            //mgpu::MergesortKeys(
            //    nvbio::device_view( d_super_suffixes ),
            //    n_suffixes,
            //    priv::string_suffix_less<SYMBOL_SIZE,string_type>( N, string ),
            //    *mgpu_context );
        #endif
            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            sufsort_time += timer.seconds();

            // compute the bwt of the block
            thrust::transform(
                d_super_suffixes.begin(),
                d_super_suffixes.begin() + n_suffixes,
                d_block_bwt.begin(),
                priv::string_bwt_functor<string_type>( N, string ) );

            // check if there is a $ sign
            const uint32 block_primary = uint32( thrust::find(
                d_block_bwt.begin(),
                d_block_bwt.begin() + n_suffixes,
                255u ) - d_block_bwt.begin() );

            if (block_primary < n_suffixes)
            {
                // keep track of the global primary position
                primary = global_suffix_offset + suffix_count + block_primary + 1u;
            }

            timer.start();

            // and copy the transformed block to the output
            priv::device_copy(
                n_suffixes,
                d_block_bwt.begin(),
                output,
                global_suffix_offset + suffix_count + 1u );

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            bwt_copy_time += timer.seconds();

            // process delayed suffixes
            if (delay_count &&
                (delay_count >= DELAY_BUFFER ||
                 subbucket_end == bucket_end))
            {
                //fprintf(stderr,"  process %u hard keys\n", delay_count);
                timer.start();

                // and sort the corresponding suffixes
                thrust::stable_sort(
                    d_delayed_suffixes.begin(),
                    d_delayed_suffixes.begin() + delay_count,
                    priv::string_suffix_less<SYMBOL_SIZE,string_type>( N, string ) );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                timer.stop();
                sufsort_time += timer.seconds();
                compression_sort.stablesort_time += timer.seconds();
                //fprintf(stderr,"    %.1f s\n", timer.seconds());

                // compute the bwt of the block
                thrust::transform(
                    d_delayed_suffixes.begin(),
                    d_delayed_suffixes.begin() + delay_count,
                    d_block_bwt.begin(),
                    priv::string_bwt_functor<string_type>( N, string ) );

                // check if there is a $ sign
                const uint32 block_primary = uint32( thrust::find(
                    d_block_bwt.begin(),
                    d_block_bwt.begin() + delay_count,
                    255u ) - d_block_bwt.begin() );

                if (block_primary < delay_count)
                {
                    // keep track of the global primary position
                    primary = global_suffix_offset + d_delayed_slots[ block_primary ] + 1u;
                }

                timer.start();

                // and scatter the resulting symbols in the proper place
                priv::device_scatter(
                    delay_count,
                    d_block_bwt.begin(),
                    d_delayed_slots.begin(),
                    output + global_suffix_offset );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                timer.stop();
                bwt_scatter_time += timer.seconds();

                delay_count = 0;
            }

            suffix_count += subbucket_size;
        }

        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"\r  sufsort : %.1fs (%.1f M suffixes/s)                     \n", sufsort_time, 1.0e-6f*float(global_suffix_offset + suffix_count)/sufsort_time) );

      #if defined(COMPRESSION_SORTING)
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    extract  : %.1fs\n", compression_sort.extract_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    r-sort   : %.1fs\n", compression_sort.radixsort_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    s-sort   : %.1fs\n", compression_sort.stablesort_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    compress : %.1fs\n", compression_sort.compress_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    compact  : %.1fs\n", compression_sort.compact_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    bwt-copy : %.1fs\n", bwt_copy_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    bwt-scat : %.1fs\n", bwt_scatter_time) );
      #endif

        global_suffix_offset += suffix_count;
    }

    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"\n    primary at %llu\n", primary) );

    // shift back all symbols following the primary
    {
        for (index_type block_begin = primary; block_begin < string_len; block_begin += max_block_size)
        {
            const index_type block_end = nvbio::min( block_begin + max_block_size, string_len );

            // copy all symbols to a temporary buffer
            priv::device_copy(
                block_end - block_begin,
                output + block_begin + 1u,
                d_block_bwt.begin(),
                uint32(0) );

            // and copy the shifted block to the output
            priv::device_copy(
                block_end - block_begin,
                d_block_bwt.begin(),
                output,
                block_begin );
        }
    }

    return primary;
}

template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type>
struct HostBWTConfig
{
    typedef typename std::iterator_traits<storage_type>::value_type word_type;

    static const uint32 WORD_BITS        = uint32( 8u * sizeof(word_type) );
    static const uint32 BUCKETING_BITS   = 16;
    static const uint32 DOLLAR_BITS      = WORD_BITS <= 32 ? 4 : 5;

    typedef typename priv::word_selector<BUCKETING_BITS>::type bucket_type;

    typedef ConcatenatedStringSet<
            PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
            uint64*>    string_set_type;

    typedef priv::HostChunkLoader<SYMBOL_SIZE,BIG_ENDIAN,storage_type>                          chunk_loader;
    typedef priv::HostStringSetRadices<string_set_type,SYMBOL_SIZE,DOLLAR_BITS,WORD_BITS>       string_set_handler;
    typedef priv::SetSuffixBucketer<SYMBOL_SIZE,BUCKETING_BITS,DOLLAR_BITS,bucket_type>         suffix_bucketer;
};

template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type>
struct DeviceBWTConfig
{
    typedef typename std::iterator_traits<storage_type>::value_type word_type;

    static const uint32 WORD_BITS        = uint32( 8u * sizeof(word_type) );
    static const uint32 BUCKETING_BITS   = 16;
    static const uint32 DOLLAR_BITS      = WORD_BITS <= 32 ? 4 : 5;

    typedef typename priv::word_selector<BUCKETING_BITS>::type bucket_type;

    typedef ConcatenatedStringSet<
            PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
            uint64*>    string_set_type;

    typedef priv::DeviceChunkLoader<SYMBOL_SIZE,BIG_ENDIAN,storage_type>                        chunk_loader;
    typedef priv::DeviceStringSetRadices<string_set_type,SYMBOL_SIZE,DOLLAR_BITS,WORD_BITS>     string_set_handler;
    typedef priv::SetSuffixBucketer<SYMBOL_SIZE,BUCKETING_BITS,DOLLAR_BITS,bucket_type>         suffix_bucketer;
};

template <typename ConfigType, uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type>
struct LargeBWTSkeleton
{
    typedef typename std::iterator_traits<storage_type>::value_type word_type;
    typedef typename ConfigType::bucket_type                        bucket_type;

    typedef ConcatenatedStringSet<
            typename PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64>::iterator,
            uint64*>    string_set_type;

    // compute the maximum sub-bucket size
    //
    static uint32 max_subbucket_size(
        const thrust::host_vector<uint32>&  h_buckets,
        const uint32                        max_super_block_size,
        const uint32                        limit)
    {
        const uint32 DOLLAR_BITS = ConfigType::DOLLAR_BITS;
        const uint32 DOLLAR_MASK = (1u << DOLLAR_BITS) - 1u;

        uint32 max_size  = 0u;
        uint32 max_index = 0u;

        // build the subbucket pointers
        for (uint32 bucket_begin = 0, bucket_end = 0; bucket_begin < h_buckets.size(); bucket_begin = bucket_end)
        {
            // grow the block of buckets until we can
            uint32 bucket_size;
            for (bucket_size = 0; (bucket_end < h_buckets.size()) && (bucket_size + h_buckets[bucket_end] <= max_super_block_size); ++bucket_end)
                bucket_size += h_buckets[bucket_end];

            // check whether a single bucket exceeds our host buffer capacity
            // TODO: if this is a short-string bucket, we could handle it with special care,
            // but it requires modifying the collecting loop to output everything directly.
            if (bucket_end == bucket_begin)
                throw nvbio::runtime_error("bucket %u contains %u strings: buffer overflow!", bucket_begin, h_buckets[bucket_begin]);

            // loop through the sub-buckets
            for (uint32 subbucket = bucket_begin; subbucket < bucket_end; ++subbucket)
            {
                // only keep track of buckets that are NOT short-string buckets
                if ((subbucket & DOLLAR_MASK) == DOLLAR_MASK)
                {
                    if (max_size < h_buckets[subbucket])
                    {
                        max_size  = h_buckets[subbucket];
                        max_index = subbucket;
                    }
                }
            }
        }
        if (max_size > limit)
            throw nvbio::runtime_error("subbucket %u contains %u strings: buffer overflow!\n  please try increasing the device memory limit to at least %u MB\n", max_index, max_size, util::divide_ri( 32u*max_size, 1024u ));
        return max_size;
    }

    // construct the sub-bucket lists
    //
    static void build_subbuckets(
        const thrust::host_vector<uint32>&  h_buckets,
        thrust::host_vector<uint32>&        h_subbuckets,
        const uint32                        max_super_block_size,
        const uint32                        max_block_size)
    {
        const uint32 DOLLAR_BITS = ConfigType::DOLLAR_BITS;
        const uint32 DOLLAR_MASK = (1u << DOLLAR_BITS) - 1u;

        // build the subbucket pointers
        for (uint32 bucket_begin = 0, bucket_end = 0; bucket_begin < h_buckets.size(); bucket_begin = bucket_end)
        {
            // grow the block of buckets until we can
            uint32 bucket_size;
            for (bucket_size = 0; (bucket_end < h_buckets.size()) && (bucket_size + h_buckets[bucket_end] <= max_super_block_size); ++bucket_end)
                bucket_size += h_buckets[bucket_end];

            // check whether a single bucket exceeds our host buffer capacity
            // TODO: if this is a short-string bucket, we could handle it with special care,
            // but it requires modifying the collecting loop to output everything directly.
            if (bucket_end == bucket_begin)
                throw nvbio::runtime_error("bucket %u contains %u strings: buffer overflow!", bucket_begin, h_buckets[bucket_begin]);

            // build the sub-buckets
            for (uint32 subbucket_begin = bucket_begin, subbucket_end = bucket_begin; subbucket_begin < bucket_end; subbucket_begin = subbucket_end)
            {
                if (h_buckets[subbucket_begin] > max_block_size)
                {
                    // if this is NOT a short-string bucket, we can't cope with it
                    if ((subbucket_begin & DOLLAR_MASK) == DOLLAR_MASK)
                        throw nvbio::runtime_error("bucket %u contains %u strings: buffer overflow!", subbucket_begin, h_buckets[subbucket_begin]);

                    // this is a short-string bucket: we can handle it with special care
                    h_subbuckets[ subbucket_end++ ] = subbucket_begin; // point to the beginning of this sub-bucket
                }
                else
                {
                    // grow the block of sub-buckets until we can
                    uint32 subbucket_size;
                    for (subbucket_size = 0; (subbucket_end < bucket_end) && (subbucket_size + h_buckets[subbucket_end] <= max_block_size); ++subbucket_end)
                    {
                        subbucket_size += h_buckets[subbucket_end];

                        h_subbuckets[ subbucket_end ] = subbucket_begin; // point to the beginning of this sub-bucket
                    }
                }
            }
        }
    }

    template <typename output_handler>
    static void enact(
        const string_set_type       string_set,
        output_handler&             output,
        BWTParams*                  params)
    {
        typedef typename ConfigType::string_set_handler     string_set_handler_type;
        typedef typename ConfigType::chunk_loader           chunk_loader_type;
        typedef typename chunk_loader_type::chunk_set_type  chunk_set_type;
        typedef typename ConfigType::suffix_bucketer        suffix_bucketer_type;

        const uint32 BUCKETING_BITS   = ConfigType::BUCKETING_BITS;
        const uint32 DOLLAR_BITS      = ConfigType::DOLLAR_BITS;
        const uint32 DOLLAR_MASK      = (1u << DOLLAR_BITS) - 1u;
        const uint32 SLICE_SIZE       = 4;

        const uint32 M = 128*1024;
        const uint32 N = string_set.size();

        const uint32 n_chunks = (N + M-1) / M;

        mgpu::ContextPtr mgpu_ctxt = mgpu::CreateCudaDevice(0); 

        suffix_bucketer_type    bucketer( mgpu_ctxt );
        chunk_loader_type       chunk;
        string_set_handler_type string_set_handler( string_set );
        cuda::CompressionSort   string_sorter( mgpu_ctxt );

        const uint32 max_super_block_size = params ?              // requires max_super_block_size*8 host memory bytes
            (params->host_memory - (128u*1024u*1024u)) / 8u :     // leave 128MB for the bucket counters
            512*1024*1024;
        uint32 max_block_size = params ?
            params->device_memory / 32 :                          // requires max_block_size*32 device memory bytes
            32*1024*1024;                                         // default: 1GB

        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  super-block-size: %.1f M\n", float(max_super_block_size)/float(1024*1024)) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        block-size: %.1f M\n", float(max_block_size)/float(1024*1024)) );
        thrust::host_vector<uint2>       h_suffixes( max_super_block_size );
        thrust::host_vector<uint2>       h_block_suffixes;
        thrust::host_vector<bucket_type> h_block_radices;
        thrust::host_vector<uint8>       h_block_bwt;

        // reuse some buffers
        thrust::device_vector<uint32>&   d_indices = bucketer.d_indices;
        thrust::device_vector<uint2>     d_bucket_suffixes;
        thrust::device_vector<uint8>     d_block_bwt;
        //thrust::device_vector<uint8>     d_temp_storage;

        // global bucket sizes
        thrust::device_vector<uint32> d_buckets( 1u << BUCKETING_BITS, 0u );

        // allocate an MGPU context
        mgpu::ContextPtr mgpu = mgpu::CreateCudaDevice(0);

        float bwt_time    = 0.0f;
        float output_time = 0.0f;

        // output the last character of each string (i.e. the symbols preceding all the dollar signs)
        const uint32 block_size = max_block_size / 4u; // this can be done in relatively small blocks
        for (uint32 block_begin = 0; block_begin < N; block_begin += block_size)
        {
            const uint32 block_end = nvbio::min( block_begin + block_size, N );

            // consume subbucket_size suffixes
            const uint32 n_suffixes = block_end - block_begin;

            Timer timer;
            timer.start();

            priv::alloc_storage( h_block_bwt, n_suffixes );
            priv::alloc_storage( d_block_bwt, n_suffixes );

            // load the BWT symbols
            string_set_handler.dollar_bwt(
                block_begin,
                block_end,
                plain_view( h_block_bwt ) );

            // copy them to the device
            thrust::copy(
                h_block_bwt.begin(),
                h_block_bwt.begin() + n_suffixes,
                d_block_bwt.begin() );

            timer.stop();
            bwt_time += timer.seconds();

            timer.start();

            // invoke the output handler
            output.process(
                n_suffixes,
                plain_view( h_block_bwt ),
                plain_view( d_block_bwt ),
                NULL,
                NULL,
                NULL );

            timer.stop();
            output_time += timer.seconds();
        }

        float load_time  = 0.0f;
        float merge_time = 0.0f;
        float count_time = 0.0f;
        Timer count_timer;
        count_timer.start();

        uint64 total_suffixes = 0;

        for (uint32 chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
        {
            const uint32 chunk_begin = chunk_id * M;
            const uint32 chunk_end   = nvbio::min( chunk_begin + M, N );

            //
            // load a chunk in device memory
            //

            Timer timer;
            timer.start();

            chunk_set_type d_chunk_set = chunk.load( string_set, chunk_begin, chunk_end );

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            load_time += timer.seconds();

            timer.start();

            // count the chunk's buckets
            bucketer.count( d_chunk_set );

            total_suffixes += bucketer.suffixes.n_suffixes;

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            count_time += timer.seconds();

            timer.start();

            // and merge them in with the global buckets
            thrust::transform(
                bucketer.d_buckets.begin(),
                bucketer.d_buckets.end(),
                d_buckets.begin(),
                d_buckets.begin(),
                thrust::plus<uint32>() );

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            merge_time += timer.seconds();
        }

        count_timer.stop();

        thrust::host_vector<uint32> h_buckets( d_buckets );
        thrust::host_vector<uint64> h_bucket_offsets( d_buckets.size() );
        thrust::host_vector<uint32> h_subbuckets( d_buckets.size() );

        const uint32 max_bucket_size = thrust::reduce(
            d_buckets.begin(),
            d_buckets.end(),
            0u,
            thrust::maximum<uint32>() );

        // scan the bucket offsets so as to have global positions
        thrust::exclusive_scan(
            thrust::make_transform_iterator( h_buckets.begin(), priv::cast_functor<uint32,uint64>() ),
            thrust::make_transform_iterator( h_buckets.end(),   priv::cast_functor<uint32,uint64>() ),
            h_bucket_offsets.begin() );

        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    max bucket size: %u\n", max_bucket_size) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    counting : %.1fs\n", count_timer.seconds() ) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      load   : %.1fs\n", load_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      merge  : %.1fs\n", merge_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      setup    : %.1fs\n", bucketer.d_setup_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        scan   : %.1fs\n", bucketer.suffixes.d_scan_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        search : %.1fs\n", bucketer.suffixes.d_search_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      count  : %.1fs\n", count_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        flatten : %.1fs\n", bucketer.d_flatten_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        sort    : %.1fs\n", bucketer.d_count_sort_time) );
        NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"        search  : %.1fs\n", bucketer.d_search_time) );

        bucketer.clear_timers();

        //
        // at this point, we have to do multiple passes through the input string set,
        // collecting in each pass as many buckets as we can fit in memory at once
        //

        float sufsort_time = 0.0f;
        float collect_time = 0.0f;
        float bin_time     = 0.0f;

        // compute the largest non-elementary bucket
        const uint32 largest_subbucket = max_subbucket_size( h_buckets, max_super_block_size, max_block_size );

        // reduce the scratchpads size if possible
        if (max_block_size > util::round_i( largest_subbucket, 32u ))
            max_block_size = util::round_i( largest_subbucket, 32u );

        // reserve memory for scratchpads
        {
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  allocating scratchpads\n" ) );

            string_set_handler.reserve( max_block_size, SLICE_SIZE );
            string_sorter.reserve( max_block_size );

            priv::alloc_storage( h_block_radices,   max_block_size );
            priv::alloc_storage( h_block_suffixes,  max_block_size );
            priv::alloc_storage( h_block_bwt,       max_block_size );
            priv::alloc_storage( d_block_bwt,       max_block_size );
            priv::alloc_storage( d_indices,         max_block_size );
            priv::alloc_storage( d_bucket_suffixes, max_block_size );

            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  allocated device memory: %.1f MB\n",
                float( bucketer.allocated_device_memory() + string_set_handler.allocated_device_memory() + string_sorter.allocated_device_memory() ) / float(1024*1024) ) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    bucketer : %.1f MB\n", float( bucketer.allocated_device_memory() ) / float(1024*1024) ) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    handler  : %.1f MB\n", float( string_set_handler.allocated_device_memory() ) / float(1024*1024) ) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    sorter   : %.1f MB\n", float( string_sorter.allocated_device_memory() ) / float(1024*1024) ) );
        }

        // now build the sub-bucket lists
        build_subbuckets(
            h_buckets,
            h_subbuckets,
            max_super_block_size,
            max_block_size );

        // build the subbucket pointers
        thrust::device_vector<uint32> d_subbuckets( h_subbuckets );

        uint64 global_suffix_offset = 0;

        for (uint32 bucket_begin = 0, bucket_end = 0; bucket_begin < h_buckets.size(); bucket_begin = bucket_end)
        {
            // grow the block of buckets until we can
            uint32 bucket_size;
            for (bucket_size = 0; (bucket_end < h_buckets.size()) && (bucket_size + h_buckets[bucket_end] <= max_super_block_size); ++bucket_end)
                bucket_size += h_buckets[bucket_end];

            uint32 suffix_count   = 0;
            uint32 string_count   = 0;
            uint32 max_suffix_len = 0;

            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  collect buckets[%u:%u] (%u suffixes)\n", bucket_begin, bucket_end, bucket_size) );
            Timer collect_timer;
            collect_timer.start();

            for (uint32 chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
            {
                const uint32 chunk_begin = chunk_id * M;
                const uint32 chunk_end   = nvbio::min( chunk_begin + M, N );
                const uint32 chunk_size  = chunk_end - chunk_begin;

                //
                // load a chunk in device memory
                //

                chunk_set_type d_chunk_set = chunk.load( string_set, chunk_begin, chunk_end );

                // collect the chunk's suffixes within the bucket range
                uint32 suffix_len;

                const uint32 n_collected = bucketer.collect(
                    d_chunk_set,
                    bucket_begin,
                    bucket_end,
                    string_count,
                    suffix_len,
                    d_subbuckets.begin(),
                    h_block_radices,
                    h_block_suffixes );

                if (suffix_count + n_collected > max_super_block_size)
                {
                    log_error(stderr,"buffer size exceeded! (%u/%u)\n", suffix_count, max_super_block_size);
                    exit(1);
                }

                Timer timer;
                timer.start();

                // dispatch each suffix to their respective bucket
                for (uint32 i = 0; i < n_collected; ++i)
                {
                    const uint2  loc    = h_block_suffixes[i];
                    const uint32 bucket = h_block_radices[i];
                    const uint64 slot   = h_bucket_offsets[bucket]++; // this could be done in parallel using atomics

                    NVBIO_CUDA_DEBUG_ASSERT(
                        slot >= global_suffix_offset,
                        slot <  global_suffix_offset + max_super_block_size,
                        "[%u] = (%u,%u) placed at %llu - %llu (%u)\n", i, loc.x, loc.y, slot, global_suffix_offset, bucket );

                    h_suffixes[ slot - global_suffix_offset ] = loc;
                }

                timer.stop();
                bin_time += timer.seconds();

                suffix_count += n_collected;
                string_count += chunk_size;

                max_suffix_len = nvbio::max( max_suffix_len, suffix_len );
            }
            collect_timer.stop();
            collect_time += collect_timer.seconds();
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"  collect : %.1fs (%.1f M suffixes/s - %.1f M scans/s)\n", collect_time, 1.0e-6f*float(global_suffix_offset + suffix_count)/collect_time, 1.0e-6f*float(total_suffixes)/collect_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    setup    : %.1fs\n", bucketer.d_setup_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      scan   : %.1fs\n", bucketer.suffixes.d_scan_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"      search : %.1fs\n", bucketer.suffixes.d_search_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    flatten  : %.1fs\n", bucketer.d_flatten_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    filter   : %.1fs\n", bucketer.d_filter_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    remap    : %.1fs\n", bucketer.d_remap_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    max      : %.1fs\n", bucketer.d_max_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    sort     : %.1fs\n", bucketer.d_collect_sort_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    copy     : %.1fs\n", bucketer.d_copy_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    bin      : %.1fs\n", bin_time) );

            //
            // at this point we have a large collection of localized suffixes to sort in h_suffixes;
            // we'll do it looping on multiple sub-buckets, on the GPU
            //

            suffix_count = 0u;

            const uint32 n_words = string_set_handler.num_words( max_suffix_len );

            for (uint32 subbucket_begin = bucket_begin, subbucket_end = bucket_begin; subbucket_begin < bucket_end; subbucket_begin = subbucket_end)
            {
                if (h_buckets[subbucket_begin] > max_block_size)
                {
                    // check if this is not a short-string bucket - it should never actually happen as we already tested for it
                    if ((subbucket_begin & DOLLAR_MASK) == DOLLAR_MASK)
                        throw nvbio::runtime_error("bucket %u contains %u strings: overflow!", subbucket_begin, h_buckets[subbucket_begin]);

                    // advance by one
                    ++subbucket_end;

                    const uint32 subbucket_size = h_buckets[subbucket_begin];

                    Timer suf_timer;
                    suf_timer.start();

                    // chop the bucket in multiple blocks
                    for (uint32 block_begin = 0; block_begin < subbucket_size; block_begin += max_block_size)
                    {
                        const uint32 block_end = nvbio::min( block_begin + max_block_size, subbucket_size );

                        // consume subbucket_size suffixes
                        const uint32 n_suffixes = block_end - block_begin;

                        // copy the host suffixes to the device
                        const uint2* h_bucket_suffixes = &h_suffixes[0] + suffix_count + block_begin;

                        // copy the suffix list to the device
                        priv::alloc_storage( d_bucket_suffixes, n_suffixes );
                        thrust::copy(
                            h_bucket_suffixes,
                            h_bucket_suffixes + n_suffixes,
                            d_bucket_suffixes.begin() );

                        // initialize the set radices
                        string_set_handler.init( n_suffixes, h_bucket_suffixes, nvbio::plain_view( d_bucket_suffixes ) );

                        Timer timer;
                        timer.start();

                        priv::alloc_storage( h_block_bwt, n_suffixes );
                        priv::alloc_storage( d_block_bwt, n_suffixes );

                        // load the BWT symbols
                        string_set_handler.bwt(
                            n_suffixes,
                            (const uint32*)NULL,
                            plain_view( h_block_bwt ),
                            plain_view( d_block_bwt ) );

                        timer.stop();
                        bwt_time += timer.seconds();

                        timer.start();

                        // invoke the output handler
                        output.process(
                            n_suffixes,
                            plain_view( h_block_bwt ),
                            plain_view( d_block_bwt ),
                            h_bucket_suffixes,
                            plain_view( d_bucket_suffixes ),
                            NULL );

                        timer.stop();
                        output_time += timer.seconds();
                    }
                    
                    suffix_count += subbucket_size;

                    suf_timer.stop();
                    sufsort_time += suf_timer.seconds();
                }
                else
                {
                    // grow the block of sub-buckets until we can
                    uint32 subbucket_size;
                    for (subbucket_size = 0; (subbucket_end < bucket_end) && (subbucket_size + h_buckets[subbucket_end] <= max_block_size); ++subbucket_end)
                        subbucket_size += h_buckets[subbucket_end];

                    NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"\r  sufsort buckets[%u:%u] (%.1f M suffixes/s)    ", subbucket_begin, subbucket_end, 1.0e-6f*float(global_suffix_offset + suffix_count)/sufsort_time) );
                    if (subbucket_size == 0)
                        continue;

                    // consume subbucket_size suffixes
                    const uint32 n_suffixes = subbucket_size;

                    try
                    {
                        // reserve enough space
                        priv::alloc_storage( d_indices, max_block_size );
                    }
                    catch (...)
                    {
                        log_error(stderr, "LargeBWTSkeleton: d_indices allocation failed!\n");
                        throw;
                    }

                    Timer suf_timer;
                    suf_timer.start();

                    // copy the host suffixes to the device
                    const uint2* h_bucket_suffixes = &h_suffixes[0] + suffix_count;

                    priv::alloc_storage( d_bucket_suffixes, n_suffixes );

                    // copy the suffix list to the device
                    thrust::copy(
                        h_bucket_suffixes,
                        h_bucket_suffixes + n_suffixes,
                        d_bucket_suffixes.begin() );

                    // initialize the set radices
                    string_set_handler.init( n_suffixes, h_bucket_suffixes, nvbio::plain_view( d_bucket_suffixes ) );

                    cuda::DiscardDelayList delay_list;

                    string_sorter.sort(
                        string_set_handler,
                        n_suffixes,
                        n_words,
                        thrust::make_counting_iterator<uint32>(0u),
                        d_indices.begin(),
                        uint32(-1),
                        delay_list,
                        SLICE_SIZE );

                    Timer timer;
                    timer.start();

                    priv::alloc_storage( h_block_bwt, n_suffixes );
                    priv::alloc_storage( d_block_bwt, n_suffixes );

                    // load the BWT symbols
                    string_set_handler.bwt(
                        n_suffixes,
                        plain_view( d_indices ),
                        plain_view( h_block_bwt ),
                        plain_view( d_block_bwt ) );

                    timer.stop();
                    bwt_time += timer.seconds();

                    timer.start();

                    // invoke the output handler
                    output.process(
                        n_suffixes,
                        plain_view( h_block_bwt ),
                        plain_view( d_block_bwt ),
                        h_bucket_suffixes,
                        plain_view( d_bucket_suffixes ),
                        plain_view( d_indices ) );

                    timer.stop();
                    output_time += timer.seconds();
                    
                    suffix_count += subbucket_size;

                    suf_timer.stop();
                    sufsort_time += suf_timer.seconds();
                }
            }
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"\r  sufsort : %.1fs (%.1f M suffixes/s)                     \n", sufsort_time, 1.0e-6f*float(global_suffix_offset + suffix_count)/sufsort_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    copy     : %.1fs\n", string_sorter.copy_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    extract  : %.1fs\n", string_sorter.extract_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    r-sort   : %.1fs\n", string_sorter.radixsort_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    compress : %.1fs\n", string_sorter.compress_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    compact  : %.1fs\n", string_sorter.compact_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    scatter  : %.1fs\n", string_sorter.scatter_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    bwt      : %.1fs\n", bwt_time) );
            NVBIO_CUDA_DEBUG_STATEMENT( log_verbose(stderr,"    output   : %.1fs\n", output_time) );

            global_suffix_offset += suffix_count;
        }
    }
};

/// Compute the bwt of a device-side string set
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type, typename output_handler>
void bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params)
{
    typedef cuda::DeviceBWTConfig<SYMBOL_SIZE,BIG_ENDIAN,storage_type> config_type;

    cuda::LargeBWTSkeleton<config_type,SYMBOL_SIZE,BIG_ENDIAN,storage_type>::enact(
        string_set,
        output,
        params );
}

} // namespace cuda

/// Compute the bwt of a host-side string set
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename word_type, typename output_handler>
void large_bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<word_type*,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params)
{
    typedef cuda::HostBWTConfig<SYMBOL_SIZE,BIG_ENDIAN,word_type*> config_type;

    cuda::LargeBWTSkeleton<config_type,SYMBOL_SIZE,BIG_ENDIAN,word_type*>::enact(
        string_set,
        output,
        params );
}

} // namespace nvbio
