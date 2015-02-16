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

#include <nvbio/basic/types.h>

namespace nvbio {

///\page bloom_filter_page Bloom Filters
///\par
/// <a href=http://en.wikipedia.org/wiki/Bloom_filter>Bloom filters</a> are a probabilistic data-structure
/// useful to <i>conservatively</i> represent membership in a set using less than 1 bit per set item.
///\par
/// NVBIO provides a simple generic class to incrementally build and work with Bloom filters
/// both on the host and the device:
///
/// - bloom_filter
///
/// \section ExampleSection Example
///
///\code
/// // populate a filter in parallel
/// template <typename bloom_filter_type, typename T>
/// __global__ void populate_kernel(const uint32 N, const T* vector, bloom_filter_type filter)
/// {
///     const uint32 i = threadIdx.x + blockIdx.x * blockDim.x;
///     if (i < N)
///         filter.insert( vector[i] );
/// }
///
/// bool bloom_test()
/// {
///     // build a set of 1M random integers
///     const uint32 N = 1000000;
///     nvbio::vector<host_tag,uint32> h_vector( N );
///
///     // fill it up
///     for (uint32 i = 0; i < N; ++i)
///         h_vector[i] = rand();
///
///     // copy it to the device
///     nvbio::vector<device_tag,uint32> d_vector = h_vector;
///
///     // construct an empty Bloom filter
///     typedef bloom_filter<2,hash_functor1,hash_functor2,uint32*> bloom_filter_type;
///
///     const uint32 filter_words = N;  // let's use 32-bits per input item;
///                                     // NOTE: this is still a lot less than 1-bit for each
///                                     // of the 4+ billion integers out there...
///
///     nvbio::vector<device_tag,uint32> d_filter_storage( filter_words, 0u );
///     bloom_filter_type d_filter(
///        filter_words * 32,
///        plain_view( d_filter_storage ) );
///
///     // and populate it
///     populate_kernel<<<util::divide_ri(N,128),128>>>( N, plain_view( d_vector ), d_filter );
///
///     // copy the filter back on the host
///     nvbio::vector<host_tag,uint32> h_filter_storage = filter_storage;
///     bloom_filter_type h_filter(
///        filter_words * 32,
///        plain_view( h_filter_storage ) );
///
///     // and now ask The Question: is 42 in there?
///     return h_filter.has( 42 );
/// }
///\endcode
///

///@addtogroup Basic
///@{

///\defgroup BloomFilterModule Bloom Filters
///
/// <a href=http://en.wikipedia.org/wiki/Bloom_filter>Bloom filters</a> are a probabilistic data-structure
/// useful to <i>conservatively</i> represent membership in a set using less than 1 bit per item.
///

///@addtogroup BloomFilterModule
///@{

///
/// atomic in-place OR binary functor used to construct a bloom_filter
///
struct inplace_or
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
    void operator() (uint32* word, const uint32 mask) const
    {
      #if defined(NVBIO_DEVICE_COMPILATION)
        atomicOr( word, mask );
      #else
        *word |= mask;
      #endif
    }
};

///
/// A Bloom filter implementation.
/// This clsss is <i>storage-free</i>, and can used both from the host and the device.
/// Constructing a Bloom filter can be done incrementally calling insert(), either sequentially
/// or in parallel.
///
/// \tparam  K              the number of hash functions, obtained as { Hash1 + i * Hash2 | i : 0, ..., K-1 }
/// \tparam  Hash1          the first hash function
/// \tparam  Hash2          the second hash function
/// \tparam  Iterator       the iterator to the internal filter storage, iterator_traits<iterator>::value_type
///                         must be a uint32
/// \tparam  OrOperator     the binary functor used to OR the filter's words with the inserted keys;
///                         NOTE: this operation must be performed atomically if the filter is constructed
///                         in parallel
///
template <
    uint32   K,         // number of hash functions { Hash1 + i * Hash2 | i : 0, ..., K-1 }
    typename Hash1,     // first hash generator function
    typename Hash2,     // second hash generator function
    typename Iterator,
    typename OrOperator = inplace_or>  // storage iterator - must dereference to uint32
struct bloom_filter
{
    /// constructor
    ///
    /// \param size         the Bloom filter's storage size, in bits
    /// \param storage      the Bloom filter's internal storage
    /// \param hash1        the first hashing function
    /// \param hash2        the second hashing function
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
    bloom_filter(
        const uint64    size,
        Iterator        storage,
        const Hash1     hash1 = Hash1(),
        const Hash2     hash2 = Hash2());

    /// insert a key
    ///
    template <typename Key>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
    void insert(const Key key, const OrOperator or_op = OrOperator());

    /// check for a key
    ///
    template <typename Key>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
    bool has(const Key key) const;

    uint64      m_size;
    Iterator    m_storage;
    Hash1       m_hash1;
    Hash2       m_hash2;
};

///@} BloomFilterModule
///@} Basic

} // namespace nvbio

#include <nvbio/basic/bloom_filter_inl.h>
