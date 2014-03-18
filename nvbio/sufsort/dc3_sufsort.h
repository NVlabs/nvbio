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

#include <nvbio/sufsort/sufsort_utils.h>
#include <nvbio/basic/string_set.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/cuda/sort.h>
#include <nvbio/basic/cuda/ldg.h>
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <mgpuhost.cuh>
#include <moderngpu.cuh>
#include <cub/cub.cuh>

namespace cufmi {
namespace cuda {

/// A functor to cast from one type into another
///
struct equal_mod3_functor
{
    typedef uint32 argument_type;
    typedef uint32 result_type;

    /// constructor
    ///
    equal_mod3_functor(const uint32 _k) : k(_k) {}

    /// return i % 3 == k
    ///
    CUFMI_FORCEINLINE CUFMI_HOST_DEVICE
    uint32 operator() (const uint32 i) const { return (i % 3 == k) ? 1u : 0u; }

    const uint32 k;
};

///
/// A sorting enactor for sorting all suffixes of a given string using a modified parallel version
/// of "Faster Suffix Sorting" by Larsson and Sadanake.
///
struct DC3SufSort
{
    /// constructor
    ///
    DC3SufSort() :
        extract_time(0.0f),
        gather_time(0.0f),
        radixsort_time(0.0f),
        segment_time(0.0f),
        compact_time(0.0f),
        inverse_time(0.0f) { m_mgpu = mgpu::CreateCudaDevice(0); }

    /// Sort all the suffixes of a given string, using a modified parallel version
    /// of "Faster Suffix Sorting" by Larsson and Sadanake.
    ///
    /// \param string_len           string length
    /// \param string               string iterator
    /// \param d_suffixes           device vector of output suffixes
    ///
    /// All the other parameters are temporary device buffers
    ///
    template <typename string_type, typename output_iterator>
    void sort(
        const typename stream_traits<string_type>::index_type   string_len,
        const string_type                                       string,
        output_iterator                                         d_suffixes);

    /// reserve enough storage for sorting n strings/suffixes
    ///
    void reserve(const uint32 n)
    {
        if (n > d_active_slots.size())
        {
            clear();

            d_12_keys.resize( n );
            d_active_slots.resize( n + 4 );
            d_sort_keys.resize( n + 4 );
            d_sort_indices.resize( n + 4 );
            d_temp_flags.resize( n + 16 );
            d_segment_flags.resize( n + 16 );
            d_segment_keys.resize( n + 4 );
        }
    }
    /// free all temporary storage
    ///
    void clear()
    {
        d_12_keys.clear();
        d_active_slots.clear();
        d_sort_keys.clear();
        d_sort_indices.clear();
        d_temp_flags.clear();
        d_segment_flags.clear();
        d_segment_keys.clear();
    }

    float extract_time;             ///< timing stats
    float gather_time;              ///< timing stats
    float radixsort_time;           ///< timing stats
    float segment_time;             ///< timing stats
    float compact_time;             ///< timing stats
    float inverse_time;             ///< timing stats

private:
    thrust::device_vector<uint32>   d_12_keys;           ///< compressed prefix keys
    thrust::device_vector<uint32>   d_sort_keys;        ///< radix-sorting keys
    thrust::device_vector<uint32>   d_sort_indices;     ///< radix-sorting indices
    thrust::device_vector<uint32>   d_active_slots;     ///< active slots vector
    thrust::device_vector<uint8>    d_segment_flags;    ///< segment head flags
    thrust::device_vector<uint8>    d_partial_flags;    ///< segment head flags
    thrust::device_vector<uint8>    d_temp_flags;       ///< segment head flags
    thrust::device_vector<uint32>   d_segment_keys;     ///< segment keys
    thrust::device_vector<uint32>   d_segment_heads;    ///< segment heads
    mgpu::ContextPtr                m_mgpu;
};

/// Sort all the suffixes of a given string, using a modified parallel version of "Faster Suffix Sorting"
/// by Larsson and Sadanake.
///
/// \param string_len           string length
/// \param string               string iterator
/// \param d_suffixes           device vector of output suffixes
///
template <typename string_type, typename output_iterator>
void DC3SufSort::sort(
    const typename stream_traits<string_type>::index_type   string_len,
    const string_type                                       string,
    output_iterator                                         d_suffixes)
{
    typedef typename stream_traits<string_type>::index_type index_type;
    const uint32 SYMBOL_SIZE = stream_traits<string_type>::SYMBOL_SIZE;

    typedef uint32 word_type;
    const uint32   WORD_BITS = uint32( 8u * sizeof(word_type) );

    const uint32 SYMBOLS_PER_WORD = 3u;

    const uint32 n_suffixes = string_len;

    // reserve temporary storage
    reserve( n_suffixes );

    // initialize the segment flags
    thrust::fill(
        d_segment_flags.begin(),
        d_segment_flags.begin() + n_suffixes,
        uint8(0u) );


    // initialize the device sorting indices
    const uint32 n_1_suffixes = uint32( thrust::copy_if(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_suffixes,
        d_sort_indices.begin(),
        equal_mod3_functor(1u) ) - d_sort_indices.begin() );

    const uint32 n_12_suffixes = uint32( thrust::copy_if(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_suffixes,
        d_sort_indices.begin() + n_1_suffixes,
        equal_mod3_functor(2u) ) - d_sort_indices.begin() );

    // initialize the active slots
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_suffixes,
        d_active_slots.begin() );

    // first pass: build the 1mod3 & 2mod3 suffixes
    {
        Timer timer;
        timer.start();

        // extract the given radix word from each of the partially sorted suffixes and merge it with the existing keys
        priv::string_suffix_word_functor<SYMBOL_SIZE,SYMBOLS_PER_WORD,WORD_BITS,DOLLAR_BITS,string_type,word_type> word_functor( string_len, string, 0u );

        thrust::copy(
            thrust::make_transform_iterator( d_sort_indices.begin(), word_functor ),
            thrust::make_transform_iterator( d_sort_indices.begin(), word_functor ) + n_12_suffixes,
            d_sort_keys.begin() );

        CUFMI_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        extract_time += timer.seconds();

        timer.start();

        cuda::SortBuffers<uint32*,uint32*> sort_buffers;
        cuda::SortEnactor                  sort_enactor;

        sort_buffers.selector  = 0;
        sort_buffers.keys[0]   = cufmi::device_view( d_sort_keys );
        sort_buffers.keys[1]   = cufmi::device_view( d_segment_keys );
        sort_buffers.values[0] = cufmi::device_view( d_active_slots );
        sort_buffers.values[1] = cufmi::device_view( d_segment_heads );

        // sort the keys together with the indices
        sort_enactor.sort( n_suffixes, sort_buffers );

        if (sort_buffers.selector)
        {
            // swap the buffers
            d_sort_keys.swap( d_segment_keys );
            d_active_slots.swap( d_segment_heads );
        }

        CUFMI_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        radixsort_time += timer.seconds();
    }

    CUFMI_CUDA_DEBUG_STATEMENT( fprintf(stderr,"    primary key compression\n") );
    {
        Timer timer;
        timer.start();

        // find out consecutive items with equal keys
        thrust::adjacent_difference(
            d_sort_keys.begin(),
            d_sort_keys.begin() + n_12_suffixes,
            d_segment_flags.begin() );

        d_segment_flags[0] = 1u;

        // and rewrite the keys using the smallest set of contiguous integers
        thrust::inclusive_scan(
            d_segment_flags.begin(),
            d_segment_flags.begin() + n_12_suffixes,
            d_segment_keys.begin() );

        CUFMI_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        segment_time += timer.seconds();

        timer.start();

        // now scatter the new keys to obtain the inverse map
        thrust::scatter(
            d_segment_keys.begin(),
            d_segment_keys.begin() + n_12_suffixes,
            d_active_slots.begin(),
            d_12_keys.begin() );

        CUFMI_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        timer.stop();
        inverse_time += timer.seconds();
    }
}

} // namespace cuda
} // namespace cufmi
