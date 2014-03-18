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
#include <nvbio/basic/string_set.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/cuda/sort.h>
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

///
/// A delay list keeping track of delayed sorting indices
///
template <typename OutputIterator>
struct DelayList
{
    DelayList(
        OutputIterator delay_output,
        OutputIterator delay_slots) :
        m_delayed( 0 ),
        m_offset( 0 ),
        m_delay_output( delay_output ),
        m_delay_slots( delay_slots ) {}

    void set_offset(const uint32 offset) { m_offset = offset; }

    template <typename InputIterator>
    void push_back(
        const uint32    n_delayed,
        InputIterator   delay_indices,
        InputIterator   delay_slots)
    {
        thrust::copy(
            delay_indices,
            delay_indices + n_delayed,
            m_delay_output + m_delayed );

        thrust::transform(
            delay_slots,
            delay_slots + n_delayed,
            m_delay_slots + m_delayed,
            priv::offset_functor( m_offset ) );

        m_delayed += n_delayed;
    }

    uint32         m_delayed;
    uint32         m_offset;
    OutputIterator m_delay_output;
    OutputIterator m_delay_slots;
};

///
/// A null delay list, discarding all delayed indices
///
struct DiscardDelayList
{
    template <typename InputIterator>
    void push_back(
        const uint32    n_delayed,
        InputIterator   delay_indices,
        InputIterator   delay_slots)
    {}
};

///
/// A sorting enactor for sorting strings using Iterative Compression Sorting - an algorithm inspired by
/// Tero Karras and Timo Aila's "Flexible Parallel Sorting through Iterative Key Compression".
///
struct CompressionSort
{
    /// constructor
    ///
    CompressionSort(mgpu::ContextPtr _mgpu) :
        extract_time(0.0f),
        radixsort_time(0.0f),
        stablesort_time(0.0f),
        compress_time(0.0f),
        compact_time(0.0f),
        copy_time(0.0f),
        scatter_time(0.0f),
        m_mgpu( _mgpu ) {}

    /// Sort a given batch of suffixes using Iterative Compression Sorting - an algorithm inspired by
    /// Tero Karras and Timo Aila's "Flexible Parallel Sorting through Iterative Key Compression".
    ///
    /// \param string_len           string length
    /// \param string               string iterator
    /// \param n_suffixes           number of suffixes to sort
    /// \param d_indices            device vector of input suffixes
    /// \param d_suffixes           device vector of output suffixes
    ///
    /// All the other parameters are temporary device buffers
    ///
    template <typename string_type, typename output_iterator, typename delay_iterator>
    void sort(
        const typename string_type::index_type  string_len,
        const string_type                       string,
        const uint32                            n_suffixes,
        output_iterator                         d_suffixes,
              uint32&                           n_delayed,
        const uint32                            delay_offset,
        delay_iterator                          delay_suffixes,
        delay_iterator                          delay_slots);

    /// Sort a given batch of strings using Iterative Compression Sorting - an algorithm inspired by
    /// Tero Karras and Timo Aila's "Flexible Parallel Sorting through Iterative Key Compression".
    ///
    /// \tparam set_type            a set of "word" strings, needs to provide the following
    ///                             interface:
    ///                             TODO
    ///
    /// \param set                  the set of items to sort
    /// \param n_strings            the number of items to sort
    /// \param d_input              device vector of input indices
    /// \param d_output             device vector of output indices
    ///
    /// All the other parameters are temporary device buffers
    ///
    template <typename set_type, typename input_iterator, typename output_iterator, typename delay_list_type>
    void sort(
              set_type                          set,
        const uint32                            n_strings,
        const uint32                            max_words,
        input_iterator                          d_input,
        output_iterator                         d_output,
        const uint32                            delay_threshold,
        delay_list_type&                        delay_list,
        const uint32                            slice_size = 8u);

    /// reserve enough storage for sorting n strings/suffixes
    ///
    void reserve(const uint32 n)
    {
        try
        {
            priv::alloc_storage( d_temp_indices,    n+4 );
            priv::alloc_storage( d_indices,         n+4 );
            priv::alloc_storage( d_keys,            n+4 );
            priv::alloc_storage( d_active_slots,    n+4 );
            priv::alloc_storage( d_segment_flags,   n+32 );
            priv::alloc_storage( d_copy_flags,      n+32 );
            priv::alloc_storage( d_temp_flags,      n+32 );
        }
        catch (...)
        {
            fprintf(stderr, "CompressionSort::reserve() : exception caught!\n");
            throw;
        }
    }

    /// return the amount of used device memory
    ///
    uint64 allocated_device_memory() const
    {
        return
            d_temp_storage.size()   * sizeof(uint8) +
            d_temp_indices.size()   * sizeof(uint32) +
            d_indices.size()        * sizeof(uint32) +
            d_keys.size()           * sizeof(uint32) +
            d_active_slots.size()   * sizeof(uint32) +
            d_segment_flags.size()  * sizeof(uint8) +
            d_copy_flags.size()     * sizeof(uint8) +
            d_temp_flags.size()     * sizeof(uint8);
    }

    float extract_time;             ///< timing stats
    float radixsort_time;           ///< timing stats
    float stablesort_time;          ///< timing stats
    float compress_time;            ///< timing stats
    float compact_time;             ///< timing stats
    float copy_time;                ///< timing stats
    float scatter_time;             ///< timing stats

private:
    thrust::device_vector<uint8>    d_temp_storage;     ///< radix-sorting indices
    thrust::device_vector<uint32>   d_temp_indices;     ///< radix-sorting indices
    thrust::device_vector<uint32>   d_indices;          ///< radix-sorting indices
    thrust::device_vector<uint32>   d_keys;             ///< radix-sorting keys
    thrust::device_vector<uint32>   d_active_slots;     ///< active slots vector
    thrust::device_vector<uint8>    d_segment_flags;    ///< segment head flags
    thrust::device_vector<uint8>    d_copy_flags;       ///< copy flags
    thrust::device_vector<uint8>    d_temp_flags;       ///< temp flags
    mgpu::ContextPtr                m_mgpu;
};

// Sort a given batch of suffixes using Iterative Compression Sorting - an algorithm inspired by
// Tero Karras and Timo Aila's "Flexible Parallel Sorting through Iterative Key Compression".
//
// \param string_len           string length
// \param string               string iterator
// \param n_suffixes           number of suffixes to sort
// \param d_suffixes           device vector of input/output suffixes
//
template <typename string_type, typename output_iterator, typename delay_iterator>
void CompressionSort::sort(
    const typename string_type::index_type  string_len,
    const string_type                       string,
    const uint32                            n_suffixes,
    output_iterator                         d_suffixes,
          uint32&                           n_delayed,
    const uint32                            delay_offset,
    delay_iterator                          delay_suffixes,
    delay_iterator                          delay_slots)
{
    typedef typename string_type::index_type index_type;
    const uint32 SYMBOL_SIZE = string_type::SYMBOL_SIZE;

    typedef uint32 word_type;
    const uint32   WORD_BITS = uint32( 8u * sizeof(uint32) );
    const uint32 DOLLAR_BITS = 4;

    // reserve temporary storage
    reserve( n_suffixes );

    // initialize the device sorting indices
    thrust::copy(
        d_suffixes,
        d_suffixes + n_suffixes,
        d_indices.begin() );

    // initialize the device sorting keys
    thrust::copy(
        thrust::make_constant_iterator<uint32>(0u),
        thrust::make_constant_iterator<uint32>(0u) + n_suffixes,
        d_keys.begin() );

    // initialize the active slots
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_suffixes,
        d_active_slots.begin() );

    // initialize the segment flags
    d_segment_flags[0] = 1u;
    thrust::fill(
        d_segment_flags.begin() + 1u,
        d_segment_flags.begin() + n_suffixes,
        uint8(0) );

    // keep track of the number of active suffixes
    uint32 n_active_suffixes = n_suffixes;

    //
    // do what is essentially an MSB radix-sort on the suffixes, word by word, using iterative key
    // compression:
    // the idea is that at each step we sort the current, say, 64-bit keys, and then "rewrite" them
    // so as to reduce their entropy to the minimum (e.g. the vector (131, 542, 542, 7184, 8192, 8192)
    // will become (0, 1, 1, 2, 3, 3)).
    // At that point, we fetch a new 32-bit radix from the strings and append it to each key, shifting
    // the old value to the high 32-bits and merging the new radix in the lowest 32.
    // And repeat, until we find out that all keys have a unique value.
    // This algorithm is a derivative of Tero Karras and Timo Aila's "Flexible Parallel Sorting through
    // Iterative Key Compression".
    //
    for (uint32 word_idx = 0; true; ++word_idx)
    {
        Timer timer;

        if (1000 * n_active_suffixes <= n_suffixes) // TODO: add a minimum pass number
        {
            /*
            timer.start();

            // if the set is small enough, switch to a comparison-based sort
            thrust::stable_sort(
                d_indices.begin(),
                d_indices.begin() + n_active_suffixes,
                priv::string_suffix_less<SYMBOL_SIZE,string_type>( string_len, string ) );

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            stablesort_time += timer.seconds();

            // scatter the partially sorted indices to the output in their proper place
            thrust::scatter(
                d_indices.begin(),
                d_indices.begin() + n_active_suffixes,
                d_active_slots.begin(),
                d_suffixes );
                */

            thrust::copy(
                d_indices.begin(),
                d_indices.begin() + n_active_suffixes,
                delay_suffixes + n_delayed );

            thrust::transform(
                d_active_slots.begin(),
                d_active_slots.begin() + n_active_suffixes,
                delay_slots + n_delayed,
                priv::offset_functor( delay_offset ) );

            n_delayed += n_active_suffixes;

            break; // bail out of the sorting loop
        }

        timer.start();

        // extract the given radix word from each of the partially sorted suffixes and merge it with the existing keys
        priv::string_suffix_word_functor<SYMBOL_SIZE,WORD_BITS,DOLLAR_BITS,string_type,uint32> word_functor( string_len, string, word_idx );

        thrust::transform(
            d_indices.begin(),
            d_indices.begin() + n_active_suffixes,
            d_keys.begin(),
            word_functor );

        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        extract_time += timer.seconds();

        timer.start();

        // build the compressed flags
        uint32* d_comp_flags = (uint32*)nvbio::device_view( d_temp_flags );
        priv::pack_flags(
            n_active_suffixes,
            nvbio::device_view( d_segment_flags ),
            d_comp_flags );

        // sort within segments
        mgpu::SegSortPairsFromFlags(
            nvbio::device_view( d_keys ),
            nvbio::device_view( d_indices ),
            n_active_suffixes,
            d_comp_flags,
            *m_mgpu );

        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        radixsort_time += timer.seconds();

        timer.start();

        // find out consecutive items with equal keys
        //
        // We can easily compute the head flags for a set of "segments" of equal keys, just by comparing each
        // of them in the sorted list with its predecessor.
        // At that point, we can isolate all segments which contain more than 1 suffix and continue sorting
        // those by themselves.
        priv::build_head_flags(
            n_active_suffixes,
            nvbio::device_view( d_keys ),
            nvbio::device_view( d_segment_flags ) );

        d_segment_flags[0]                 = 1u; // make sure the first flag is a 1
        d_segment_flags[n_active_suffixes] = 1u; // and add a sentinel

        // perform a scan to "compress" the keys in place, removing holes between them and reducing their entropy;
        // this operation will produce a 1-based vector of contiguous values of the kind (1, 1, 2, 3, 3, 3, ... )
        priv::inclusive_scan(
            n_active_suffixes,
            thrust::make_transform_iterator( d_segment_flags.begin(), priv::cast_functor<uint8,uint32>() ),
            d_keys.begin(),
            thrust::plus<uint32>(),
            d_temp_storage );

        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        compress_time += timer.seconds();

        const uint32 n_segments = d_keys[ n_active_suffixes - 1u ];
        //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    segments: %u/%u, at pass %u\n", n_segments, n_active_suffixes, word_idx) );

        if (n_segments == n_active_suffixes) // we are done!
        {
            //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    sorted in %u passes\n", word_idx) );

            // scatter the partially sorted indices to the output in their proper place
            thrust::scatter(
                d_indices.begin(),
                d_indices.begin() + n_active_suffixes,
                d_active_slots.begin(),
                d_suffixes );

            break; // bail out of the sorting loop
        }

    #define KEY_PRUNING
    #if defined(KEY_PRUNING)
        //
        // extract all the strings from segments with more than 1 element: in order to do this we want to mark
        // all the elements that need to be copied. These are exactly all the 1's followed by a 0, and all the
        // 0's. Conversely, all the 1's followed by a 1 don't need to copied.
        // i.e. we want to transform the flags:
        //  (1,1,0,0,0,1,1,1,0,1) in the vector:
        //  (0,1,1,1,1,0,0,1,1,0)
        // This can be done again looking at neighbours, this time with a custom binary-predicate implementing
        // the above logic.
        // As thrust::adjacent_difference is right-aligned, we'll obtain the following situation instead:
        //  segment-flags: (1,1,0,0,0,1,1,1,0,1,[1])     (where [1] is a manually inserted sentinel value)
        //  copy-flags:  (1,0,1,1,1,1,0,0,1,1,0,[.])
        // where we have added a sentinel value to the first vector, and will find our results starting
        // at position one in the output
        //
        thrust::adjacent_difference(
            d_segment_flags.begin(),
            d_segment_flags.begin() + n_active_suffixes+1u,
            d_copy_flags.begin(),
            priv::remove_singletons() );

        const uint32 n_partials = thrust::transform_reduce(
            d_copy_flags.begin() + 1u,
            d_copy_flags.begin() + 1u + n_active_suffixes,
            priv::cast_functor<uint8,uint32>(),
            0u,
            thrust::plus<uint32>() );

        // check if the number of "unique" keys is small enough to justify reducing the active set
        //if (2u*n_segments >= n_active_suffixes)
        if (2u*n_partials <= n_active_suffixes)
        {
            //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    segments: %.3f%%, at pass %u\n", 100.0f*float(n_segments)/float(n_suffixes), word_idx) );

            timer.start();

            // scatter the partially sorted indices to the output in their proper place
            thrust::scatter(
                d_indices.begin(),
                d_indices.begin() + n_active_suffixes,
                d_active_slots.begin(),
                d_suffixes );

            thrust::device_vector<uint32>& d_temp_indices = d_keys;

            // now keep only the indices we are interested in
            priv::copy_if(
                n_active_suffixes,
                d_indices.begin(),
                d_copy_flags.begin() + 1u,
                d_temp_indices.begin(),
                d_temp_storage );

            d_indices.swap( d_temp_indices );

            // as well as their slots
            priv::copy_if(
                n_active_suffixes,
                d_active_slots.begin(),
                d_copy_flags.begin() + 1u,
                d_temp_indices.begin(),
                d_temp_storage );

            d_active_slots.swap( d_temp_indices );

            // and the segment flags
            priv::copy_if(
                n_active_suffixes,
                d_segment_flags.begin(),
                d_copy_flags.begin() + 1u,
                d_temp_flags.begin(),
                d_temp_storage );

            d_segment_flags.swap( d_temp_flags );

            // overwrite the number of active suffixes
            n_active_suffixes = n_partials;

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            compact_time += timer.seconds();

            //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    active: %.3f%% in %.3f%% segments, at pass %u\n", 100.0f*float(n_active_suffixes)/float(n_suffixes), 100.0f*float(n_segments)/float(n_suffixes), word_idx) );
        }
      #endif // if defined(KEY_PRUNING)
    }
}

// Sort a given batch of strings using Iterative Compression Sorting - an algorithm inspired by
// Tero Karras and Timo Aila's "Flexible Parallel Sorting through Iterative Key Compression".
//
// \param set                  the set of items to sort
// \param n_strings            the number of items to sort
// \param d_output             device vector of output indices
//
// All the other parameters are temporary device buffers
//
template <typename set_type, typename input_iterator, typename output_iterator, typename delay_list_type>
void CompressionSort::sort(
          set_type                          set,
    const uint32                            n_strings,
    const uint32                            max_words,
    input_iterator                          d_input,
    output_iterator                         d_output,
    const uint32                            delay_threshold,
    delay_list_type&                        delay_list,
    const uint32                            slice_size)
{
    typedef uint32 index_type;

    try
    {
        // reserve temporary storage
        reserve( n_strings );

        // initialize the device sorting indices
        thrust::copy(
            d_input,
            d_input + n_strings,
            d_indices.begin() );

        // initialize the active slots
        thrust::copy(
            thrust::make_counting_iterator<uint32>(0u),
            thrust::make_counting_iterator<uint32>(0u) + n_strings,
            d_active_slots.begin() );

        // initialize the segment flags
        d_segment_flags[0] = 1u;
        thrust::fill(
            d_segment_flags.begin() + 1u,
            d_segment_flags.begin() + n_strings,
            uint8(0) );

        // keep track of the number of active suffixes
        uint32 n_active_strings = n_strings;

        //
        // do what is essentially an MSB radix-sort on the suffixes, word by word, using iterative key
        // compression:
        // the idea is that at each step we sort the current, say, 64-bit keys, and then "rewrite" them
        // so as to reduce their entropy to the minimum (e.g. the vector (131, 542, 542, 7184, 8192, 8192)
        // will become (0, 1, 1, 2, 3, 3)).
        // At that point, we fetch a new 32-bit radix from the strings and append it to each key, shifting
        // the old value to the high 32-bits and merging the new radix in the lowest 32.
        // And repeat, until we find out that all keys have a unique value.
        // This algorithm is a derivative of Tero Karras and Timo Aila's "Flexible Parallel Sorting through
        // Iterative Key Compression".
        //
        for (uint32 word_block_begin = 0; word_block_begin < max_words; word_block_begin += slice_size)
        {
            const uint32 word_block_end = nvbio::min( word_block_begin + slice_size, max_words );

            Timer timer;
            timer.start();

            // extract the given radix word from each of the partially sorted suffixes on the host
            set.init_slice(
                n_active_strings,
                n_active_strings == n_strings ? (const uint32*)NULL : plain_view( d_active_slots ),
                word_block_begin,
                word_block_end );

            NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
            timer.stop();
            extract_time += timer.seconds();

            // do what is essentially an MSB sort on the suffixes
            for (uint32 word_idx = word_block_begin; word_idx < word_block_end; ++word_idx)
            {
                if (word_idx > delay_threshold && 1000 * n_active_strings <= n_strings) // TODO: add a minimum pass number
                {
                    delay_list.push_back(
                        n_active_strings,
                        d_indices.begin(),
                        d_active_slots.begin() );

                    return; // bail out of the sorting loop
                }

                timer.start();

                set.extract(
                    n_active_strings,
                    plain_view( d_active_slots ),
                    word_idx,
                    word_block_begin,
                    word_block_end,
                    plain_view( d_temp_indices ) );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                timer.stop();
                extract_time += timer.seconds();

                timer.start();

                // get the radices in proper order
                thrust::gather(
                    d_indices.begin(),
                    d_indices.begin() + n_active_strings,
                    d_temp_indices.begin(),
                    d_keys.begin() );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                timer.stop();
                copy_time += timer.seconds();

                timer.start();

                // build the compressed flags
                uint32* d_comp_flags = (uint32*)nvbio::device_view( d_temp_flags );
                priv::pack_flags(
                    n_active_strings,
                    nvbio::device_view( d_segment_flags ),
                    d_comp_flags );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                cuda::check_error("CompressionSort::sort() : pack_flags");

                // sort within segments
                mgpu::SegSortPairsFromFlags(
                    nvbio::device_view( d_keys ),
                    nvbio::device_view( d_indices ),
                    n_active_strings,
                    d_comp_flags,
                    *m_mgpu );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                cuda::check_error("CompressionSort::sort() : seg_sort");
                timer.stop();
                radixsort_time += timer.seconds();

                timer.start();

                // find out consecutive items with equal keys
                //
                // We can easily compute the head flags for a set of "segments" of equal keys, just by comparing each
                // of them in the sorted list with its predecessor.
                // At that point, we can isolate all segments which contain more than 1 suffix and continue sorting
                // those by themselves.
                priv::build_head_flags(
                    n_active_strings,
                    nvbio::device_view( d_keys ),
                    nvbio::device_view( d_segment_flags ) );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                cuda::check_error("CompressionSort::sort() : build_head_flags");

                d_segment_flags[0]                = 1u; // make sure the first flag is a 1
                d_segment_flags[n_active_strings] = 1u; // and add a sentinel

                // perform a scan to "compress" the keys in place, removing holes between them and reducing their entropy;
                // this operation will produce a 1-based vector of contiguous values of the kind (1, 1, 2, 3, 3, 3, ... )
                priv::inclusive_scan(
                    n_active_strings,
                    thrust::make_transform_iterator( d_segment_flags.begin(), priv::cast_functor<uint8,uint32>() ),
                    d_keys.begin(),
                    thrust::plus<uint32>(),
                    d_temp_storage );

                NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                timer.stop();
                compress_time += timer.seconds();

                const uint32 n_segments = d_keys[ n_active_strings - 1u ];
                //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    segments: %u/%u, at pass %u\n", n_segments, n_active_strings, word_idx) );

                if (n_segments == n_active_strings ||
                    word_idx+1 == max_words) // we are done!
                {
                    timer.start();

                    if (n_active_strings == n_strings)
                    {
                        // copy the fully sorted indices to the output in their proper place
                        thrust::copy(
                            d_indices.begin(),
                            d_indices.begin() + n_active_strings,
                            d_output );

                    }
                    else
                    {
                        // scatter the partially sorted indices to the output in their proper place
                        thrust::scatter(
                            d_indices.begin(),
                            d_indices.begin() + n_active_strings,
                            d_active_slots.begin(),
                            d_output );
                    }
                    NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                    timer.stop();
                    scatter_time += timer.seconds();
                    return; // bail out of the sorting loop
                }
            }

            if (word_block_end < max_words)
            {
                //
                // extract all the strings from segments with more than 1 element: in order to do this we want to mark
                // all the elements that need to be copied. These are exactly all the 1's followed by a 0, and all the
                // 0's. Conversely, all the 1's followed by a 1 don't need to copied.
                // i.e. we want to transform the flags:
                //  (1,1,0,0,0,1,1,1,0,1) in the vector:
                //  (0,1,1,1,1,0,0,1,1,0)
                // This can be done again looking at neighbours, this time with a custom binary-predicate implementing
                // the above logic.
                // As thrust::adjacent_difference is right-aligned, we'll obtain the following situation instead:
                //  segment-flags: (1,1,0,0,0,1,1,1,0,1,[1])     (where [1] is a manually inserted sentinel value)
                //  copy-flags:  (1,0,1,1,1,1,0,0,1,1,0,[.])
                // where we have added a sentinel value to the first vector, and will find our results starting
                // at position one in the output
                //
                thrust::adjacent_difference(
                    d_segment_flags.begin(),
                    d_segment_flags.begin() + n_active_strings+1u,
                    d_copy_flags.begin(),
                    priv::remove_singletons() );


                const uint32 n_partials = priv::reduce(
                    n_active_strings,
                    thrust::make_transform_iterator( d_copy_flags.begin() + 1u, priv::cast_functor<uint8,uint32>() ),
                    thrust::plus<uint32>(),
                    d_temp_storage );

                // check if the number of "unique" keys is small enough to justify reducing the active set
                //if (2u*n_segments >= n_active_strings)
                if (2u*n_partials <= n_active_strings)
                {
                    //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    segments: %.3f%%, at pass %u\n", 100.0f*float(n_segments)/float(n_strings), word_idx) );

                    timer.start();

                    // scatter the partially sorted indices to the output in their proper place
                    thrust::scatter(
                        d_indices.begin(),
                        d_indices.begin() + n_active_strings,
                        d_active_slots.begin(),
                        d_output );

                    // check whether we are done sorting
                    if (n_partials == 0)
                    {
                        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                        timer.stop();
                        compact_time += timer.seconds();
                        return;
                    }

                    thrust::device_vector<uint32>& d_temp_indices = d_keys;

                    // now keep only the indices we are interested in
                    priv::copy_if(
                        n_active_strings,
                        d_indices.begin(),
                        d_copy_flags.begin() + 1u,
                        d_temp_indices.begin(),
                        d_temp_storage );

                    d_indices.swap( d_temp_indices );

                    // as well as their slots
                    priv::copy_if(
                        n_active_strings,
                        d_active_slots.begin(),
                        d_copy_flags.begin() + 1u,
                        d_temp_indices.begin(),
                        d_temp_storage );

                    d_active_slots.swap( d_temp_indices );

                    // and the segment flags
                    priv::copy_if(
                        n_active_strings,
                        d_segment_flags.begin(),
                        d_copy_flags.begin() + 1u,
                        d_temp_flags.begin(),
                        d_temp_storage );

                    d_segment_flags.swap( d_temp_flags );

                    // overwrite the number of active suffixes
                    n_active_strings = n_partials;

                    NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
                    timer.stop();
                    compact_time += timer.seconds();

                    //NVBIO_CUDA_DEBUG_STATEMENT( fprintf(stderr,"\n    active: %.3f%% in %.3f%% segments, at pass %u\n", 100.0f*float(n_active_strings)/float(n_strings), 100.0f*float(n_segments)/float(n_strings), word_idx) );
                }
            }
        }
    }
    catch (cuda_error& error)
    {
        fprintf(stderr, "CompressionSort::sort() : cuda_error caught!\n  %s\n", error.what());
        throw error;
    }
    catch (...)
    {
        fprintf(stderr, "CompressionSort::sort() : exception caught!\n");
        throw;
    }
}

} // namespace cuda
} // namespace nvbio
