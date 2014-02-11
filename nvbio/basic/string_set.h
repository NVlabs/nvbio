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

#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/vector_wrapper.h>
#include <nvbio/basic/strided_iterator.h>
#include <nvbio/basic/cached_iterator.h>


namespace nvbio {

///\page string_sets_page String Sets
///
/// A string set is a collection of strings. As there's many ways to encode a string,
/// there's even more ways to represent a string set.
/// For example, one might want to store the strings into a single concatenated array,
/// and store offsets to the beginning of each one.
/// Or he might want to build a small string set out of a sparse subset of a big string
/// (for example, to carve a few isolated regions out of a genome).
/// And the support string might be either encoded with ascii characters, or packed
/// using a few bits per symbol.
///
/// \section AtAGlanceSection At a Glance
///
/// This module provides a few generic adaptors that can be "configured" by means of
/// the underlying template iterators.
/// The philosohpy behind all of these containers is that they are just <i>shallow representations</i>,
/// holding no storage. Hence they can be used both in host and device code.
///
/// - ConcatenatedStringSet
/// - SparseStringSet
/// - StridedPackedStringSet
/// - StridedStringSet
///
/// Furthermore, the module provides efficient generic copy() (resp. cuda::copy()) implementations to copy
/// a given host (resp. device) string set from a given layout into another with a different layout.
///
/// \section StringSetInterface Interface
///
/// String sets are generic, interchangeable containers that expose the same interface:
///\code
/// interface StringSet
/// {
///     // specify the type of the strings
///     typedef ... string_type;
///
///     // return the size of the set
///     uint32 size() const;
///
///     // return the i-th string in the set
///     string_type operator[] (const uint32) [const];
/// }
///\endcode
///
/// \section TechnicalOverviewSection Technical Overview
///
/// For a detailed description of all classes and functions see the \ref StringSetsModule module documentation.
///

///@addtogroup Basic
///@{

///\defgroup StringSetsModule String Sets
///
/// This module defines various types of string sets which vary for the internal representation.
/// For a deeper explanation, see the \ref string_sets_page page.
///

///@addtogroup StringSetsModule
///@{

struct concatenated_string_set_tag {};
struct sparse_string_set_tag {};
struct strided_string_set_tag {};
struct strided_packed_string_set_tag {};

///
/// A "flat" collection of strings that are concatenated together into
/// a single one, and their starting points are given by a single offset vector.
///
/// Here's an example defining a simple concatenated string set:
///
///\code
/// // pack a few strings into a concatenated buffer
/// const uint32 n_strings = 3;
/// const char* strings = { "abc", "defghi", "lmno" };
///
///
/// thrust::host_vector<uint32> offsets_storage( n_strings+1 );
/// thrust::host_vector<char>   string_storage( strlen( strings[0] ) +
///                                             strlen( strings[1] ) +
///                                             strlen( strings[2] ) );
///
/// typedef ConcatenatedStringSet<char*, uint32*> string_set_type;
///
/// // build the string set
/// string_set_type string_set(
///     n_strings,
///     char*,
///     plain_view( offsets_storage ) );
///
/// // setup the offsets, note we need to place a sentinel
/// offsets_storage[0] = 0;
/// for (uint32 i = 0; i < n_reads; ++i)
///     offsets_storage[i+1] += strlen( strings[i] );
///
/// // and now we can conveniently access the i-th string of the set
/// for (uint32 i = 0; i < n_reads; ++i)
/// {
///     string_set_type::string_type string = string_set[i];
///
///     // and fill them in
///     for (uint32 j = 0; j < string.length(); ++j)
///         string[j] = strings[i][j];
/// }
///\endcode
///
/// or even a packed one:
///
/// \anchor ConcatenatedStringSetExample
///\code
/// // pack 1000 x 100bp reads into a single concatenated buffer
/// const uint32 n_reads    = 1000;
/// const uint32 read_len   = 100;
/// const uint32 n_words    = util::divide_ri( n_reads * read_len * 2, 32 );    // we can fit 16 bps in each word
///
/// thrust::host_vector<uint32> offsets_storage( n_reads+1 );
/// thrust::host_vector<uint32> string_storage( n_words );
/// typedef PackedStream<uint32*, uint8, 2u, false> packed_stream_type;
/// typedef packed_stream::iterator                 packed_iterator;
/// typedef uint32*                                 offsets_iterator;
/// typedef ConcatenatedStringSet<packed_iterator, offsets_iterator> packed_string_set;
///
/// packed_stream_type packed_stream( plain_view( string_storage ) );
///
/// // build the string set
/// packed_string_set string_set(
///     n_reads,
///     packed_stream.begin(),
///     plain_view( offsets_storage ) );
///
/// // setup the offsets, note we need to place a sentinel
/// for (uint32 i = 0; i < n_reads+1; ++i)
///     offsets_storage[i] = i * read_len;
///
/// // and now we can conveniently access the i-th string of the set
/// for (uint32 i = 0; i < n_reads; ++i)
/// {
///     packed_string_set::string_type string = string_set[i];
///
///     // and fill them in
///     for (uint32 j = 0; j < string.length(); ++j)
///         string[j] = ...
/// }
///\endcode
///
template <typename StringIterator, typename OffsetIterator>
struct ConcatenatedStringSet
{
    typedef concatenated_string_set_tag                                 string_set_tag;
    typedef typename std::iterator_traits<StringIterator>::value_type   symbol_type;
    typedef vector_wrapper<StringIterator>                              string_type;
    typedef StringIterator                                              symbol_iterator;
    typedef OffsetIterator                                              offset_iterator;

    /// default constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ConcatenatedStringSet() {}

    /// constructor
    ///
    /// \param size             set size
    /// \param string           flat string iterator
    /// \param offsets          string offsets in the flat string array, must contain size+1 entries
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ConcatenatedStringSet(
        const uint32         size,
        const StringIterator string,
        const OffsetIterator offsets) :
        m_size( size ),
        m_string( string ),
        m_offsets( offsets ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const typename std::iterator_traits<OffsetIterator>::value_type offset = m_offsets[i];

        return string_type(
            m_offsets[i+1] - offset,
            m_string + offset );
    }

    /// return the base string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_iterator base_string() const { return m_string; }

    /// return the offset vector
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    offset_iterator offsets() const { return m_offsets; }

private:
    uint32         m_size;
    StringIterator m_string;
    OffsetIterator m_offsets;
};

///\relates ConcatenatedStringSet
///
/// A utility function to make a ConcatenatedStringSet
///
template <typename StringIterator, typename OffsetIterator>
ConcatenatedStringSet<StringIterator,OffsetIterator> make_concatenated_string_set(
    const uint32         size,
    const StringIterator string,
    const OffsetIterator offsets)
{
    return ConcatenatedStringSet<StringIterator,OffsetIterator>(
        size,
        string,
        offsets );
}

///
/// A sparse collection of strings that are stored as ranges of a larger
/// string, and their starting points and end points are given by a range vector.
///
/// \tparam StringIterator      base string support
/// \tparam RangeIterator       an iterator definining a set of ranges, whose value_type
///                             must be <i>uint2</i>.
///
/// Assume you have a large packed-DNA genome and have identified a few isolated regions
/// of importance that you want to analyze. With the following container you can
/// easily represent them:
///
/// \anchor SparseStringSetExample
///\code
/// void analyze_regions(
///     const thrust::device_vector<uint32>& genome_storage,
///     const thrust::device_vector<uint2>&  regions)
/// {
///     typedef PackedStream<const uint32*, uint8, 2u, true>        packed_stream_type;
///     typedef packed_stream::iterator                             packed_iterator;
///     typedef const uint2*                                        ranges_iterator;
///     typedef SparseStringSet<packed_iterator, ranges_iterator>   sparse_string_set;
///
///     packed_stream_type packed_stream( plain_view( genome_storage ) );
///
///     // build the string set
///     sparse_string_set string_set(
///         regions.size(),
///         packed_stream.begin(),
///         plain_view( regions ) );
///
///     // work with the string set
///     ...
/// }
///\endcode
///
template <typename StringIterator, typename RangeIterator>
struct SparseStringSet
{
    typedef sparse_string_set_tag                                       string_set_tag;
    typedef typename std::iterator_traits<StringIterator>::value_type   symbol_type;
    typedef vector_wrapper<StringIterator>                              string_type;
    typedef StringIterator                                              symbol_iterator;
    typedef RangeIterator                                               range_iterator;

    /// default constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SparseStringSet() {}

    /// constructor
    ///
    /// \param size             set size
    /// \param string           flat string iterator
    /// \param ranges           string ranges in the flat string array
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SparseStringSet(
        const uint32         size,
        const StringIterator string,
        const RangeIterator  ranges) :
        m_size( size ),
        m_string( string ),
        m_ranges( ranges ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const uint2 range = m_ranges[i];

        return string_type(
            range.y - range.x,
            m_string + range.x );
    }

    /// return the base string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_iterator base_string() const { return m_string; }

    /// return the offset vector
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    range_iterator ranges() const { return m_ranges; }

private:
    uint32         m_size;
    StringIterator m_string;
    RangeIterator  m_ranges;
};

///\relates SparseStringSet
///
/// A utility function to make a SparseStringSet
///
template <typename StringIterator, typename RangeIterator>
SparseStringSet<StringIterator,RangeIterator> make_sparse_string_set(
    const uint32         size,
    const StringIterator string,
    const RangeIterator offsets)
{
    return SparseStringSet<StringIterator,RangeIterator>(
        size,
        string,
        offsets );
}

///
/// A collection of packed strings stored in strided vectors of words.
/// i.e. if the stride is n, the i-th string s_i is stored in the words w(i + 0), w(i + n), w(i + 2n), ...
///
/// <table>
/// <tr><td><b>s0</b></td>     <td><b>s1</b></td>       <td><b>s2</b></td>      <td>...</td></tr>
/// <tr><td>w(0 + 0)</td>      <td>w(1 + 0)</td>        <td>w(2 + 0)</td>       <td>...</td></tr>
/// <tr><td>w(0 + n)</td>      <td>w(1 + n)</td>        <td>w(2 + n)</td>       <td>...</td></tr>
/// <tr><td>w(0 + 2n)</td>     <td>w(1 + 2n)</td>       <td>w(2 + 2n)</td>      <td>...</td></tr>
/// </table>
///
/// This representation can be convenient for kernels where the i-th thread (modulo the grid-size)
/// operates on the i-th string and the character accesses are in-sync, as in this case all the 
/// memory accesses will be coalesced.
/// \n\n
/// Note that the number of words must be at least <i>n</i> times the number of words needed
/// to store the longest string in the set.
///
template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
struct StridedPackedStringSet
{
    typedef strided_packed_string_set_tag                                           string_set_tag;
    typedef SymbolType                                                              symbol_type;

    typedef StreamIterator                                                              stream_iterator;
    typedef strided_iterator<StreamIterator>                                            strided_stream_iterator;
    typedef PackedStream<strided_stream_iterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> packed_stream_type;
    typedef typename packed_stream_type::iterator                                       packed_stream_iterator;
    typedef vector_wrapper<packed_stream_type>                                          string_type;
    typedef LengthIterator                                                              length_iterator;

    static const uint32 SYMBOL_SIZE  = SYMBOL_SIZE_T;
    static const bool   BIG_ENDIAN   = BIG_ENDIAN_T;

    /// default constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StridedPackedStringSet() {}

    /// constructor
    ///
    /// \param size             set size
    /// \param stride           set stride
    /// \param string           flat string iterator
    /// \param lengths          string lengths
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StridedPackedStringSet(
        const uint32         size,
        const uint32         stride,
        const StreamIterator stream,
        const LengthIterator lengths) :
        m_size( size ),
        m_stride( stride ),
        m_stream( stream ),
        m_lengths( lengths ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// stride
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 stride() const { return m_stride; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const uint32 length = m_lengths[i];

        const strided_stream_iterator base_iterator( m_stream + i, m_stride );
        const packed_stream_type packed_stream( base_iterator );

        return string_type(
            length,
            packed_stream );
    }

    /// return the base string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    stream_iterator base_stream() const { return m_stream; }

    /// return the length vector
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    length_iterator lengths() const { return m_lengths; }

private:
    uint32         m_size;
    uint32         m_stride;
    StreamIterator m_stream;
    LengthIterator m_lengths;
};

///
/// A collection of strings stored in strided fashion.
/// i.e. if the stride is n, the i-th string s_i is stored in the symbols s(i + 0), s(i + n), s(i + 2n), ...
///
/// <table>
/// <tr><td><b>s0</b></td>     <td><b>s1</b></td>       <td><b>s2</b></td>      <td>...</td></tr>
/// <tr><td>s(0 + 0)</td>      <td>s(1 + 0)</td>        <td>s(2 + 0)</td>       <td>...</td></tr>
/// <tr><td>s(0 + n)</td>      <td>s(1 + n)</td>        <td>s(2 + n)</td>       <td>...</td></tr>
/// <tr><td>s(0 + 2n)</td>     <td>s(1 + 2n)</td>       <td>s(2 + 2n)</td>      <td>...</td></tr>
/// </table>
///
/// This representation can be convenient for kernels where the i-th thread (modulo the grid-size)
/// operates on the i-th string and the character accesses are in-sync, as in this case all the 
/// memory accesses will be coalesced.
/// \n\n
/// Note that the number of symbols must be at least <i>n</i> times the length of the longest
/// string in the set.
///
template <
    typename StringIterator,
    typename LengthIterator>
struct StridedStringSet
{
    typedef strided_string_set_tag                                                  string_set_tag;
    typedef typename std::iterator_traits<StringIterator>::value_type               symbol_type;

    typedef StringIterator                                                          symbol_iterator;
    typedef strided_iterator<StringIterator>                                        strided_symbol_iterator;
    typedef vector_wrapper<strided_symbol_iterator>                                 string_type;
    typedef LengthIterator                                                          length_iterator;

    /// default constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StridedStringSet() {}

    /// constructor
    ///
    /// \param size             set size
    /// \param stride           set stride
    /// \param string           flat string iterator
    /// \param lengths          string lengths
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StridedStringSet(
        const uint32         size,
        const uint32         stride,
        const StringIterator string,
        const LengthIterator lengths) :
        m_size( size ),
        m_stride( stride ),
        m_string( string ),
        m_lengths( lengths ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// stride
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 stride() const { return m_stride; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const uint32 length = m_lengths[i];

        const strided_symbol_iterator base_iterator( m_string + i, m_stride );

        return string_type(
            length,
            base_iterator );
    }

    /// return the base string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_iterator base_string() const { return m_string; }

    /// return the length vector
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    length_iterator lengths() const { return m_lengths; }

private:
    uint32         m_size;
    uint32         m_stride;
    StringIterator m_string;
    LengthIterator m_lengths;
};

///@} StringSetsModule
///@} Basic

namespace cuda {

///@addtogroup Basic
///@{

///@addtogroup StringSetsModule
///@{

/// copy a generic string set into a concatenated one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StringIterator,
    typename OffsetIterator>
void copy(
    const InStringSet&                                          in_string_set,
          ConcatenatedStringSet<StringIterator,OffsetIterator>& out_string_set);

/// copy a generic string set into a strided one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StringIterator,
    typename LengthIterator>
void copy(
    const InStringSet&                                      in_string_set,
          StridedStringSet<StringIterator,LengthIterator>&  out_string_set);

/// copy a generic string set into a strided-packed one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
void copy(
    const InStringSet&                                                                                  in_string_set,
          StridedPackedStringSet<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T,LengthIterator>&  out_string_set);

///@} StringSetsModule
///@} Basic

} // namespace cuda

///@addtogroup Basic
///@{

///@addtogroup StringSetsModule
///@{

/// copy a generic string set into a concatenated one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StringIterator,
    typename OffsetIterator>
void copy(
    const InStringSet&                                          in_string_set,
          ConcatenatedStringSet<StringIterator,OffsetIterator>& out_string_set);

/// copy a generic string set into a strided one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StringIterator,
    typename LengthIterator>
void copy(
    const InStringSet&                                      in_string_set,
          StridedStringSet<StringIterator,LengthIterator>&  out_string_set);

/// copy a generic string set into a strided-packed one
///
/// \param in_string_set        input string set
/// \param out_string_set       output string set
///
template <
    typename InStringSet,
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
void copy(
    const InStringSet&                                                                                  in_string_set,
          StridedPackedStringSet<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T,LengthIterator>&  out_string_set);

///@} StringSetsModule
///@} Basic


template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator,
    typename value_type>
struct CachedPackedConcatStringSet
{
};

template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
struct CachedPackedConcatStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    uint4>
{
    typedef const_cached_iterator<StreamIterator>                                       cached_base_iterator;
    typedef uint4_as_uint32_iterator<cached_base_iterator>                              uint4_iterator;
    typedef const_cached_iterator<uint4_iterator>                                       cached_stream_iterator;
    typedef PackedStream<cached_stream_iterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>  cached_packed_stream_type;
    typedef typename cached_packed_stream_type::iterator                                cached_packed_stream_iterator;
    typedef ConcatenatedStringSet<cached_packed_stream_iterator,LengthIterator>         cached_string_set;

    static cached_string_set make(
        const ConcatenatedStringSet<
            PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
            LengthIterator> string_set)
    {
        cached_packed_stream_type cached_packed_stream(
            cached_stream_iterator(
                uint4_iterator( cached_base_iterator( string_set.base_string().container().stream() ) )
                )
            );

        return cached_string_set(
            string_set.size(),
            cached_packed_stream.begin(),
            string_set.offsets() );
    }
};

template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
struct CachedPackedConcatStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    uint32>
{
    typedef const_cached_iterator<StreamIterator>                                       cached_stream_iterator;
    typedef PackedStream<cached_stream_iterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>  cached_packed_stream_type;
    typedef typename cached_packed_stream_type::iterator                                cached_packed_stream_iterator;
    typedef ConcatenatedStringSet<cached_packed_stream_iterator,LengthIterator>         cached_string_set;

    static cached_string_set make(
        const ConcatenatedStringSet<
            PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
            LengthIterator> string_set)
    {
        cached_packed_stream_type cached_packed_stream(
            cached_stream_iterator(
                    string_set.base_string().container().stream() )
            );

        return cached_string_set(
            string_set.size(),
            cached_packed_stream.begin(),
            string_set.offsets() );
    }
};

template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator,
    typename value_type>
struct CachedPackedSparseStringSet
{
};

template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
struct CachedPackedSparseStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    uint4>
{
    typedef const_cached_iterator<StreamIterator>                                       cached_base_iterator;
    typedef uint4_as_uint32_iterator<cached_base_iterator>                              uint4_iterator;
    typedef const_cached_iterator<uint4_iterator>                                       cached_stream_iterator;
    typedef PackedStream<cached_stream_iterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>  cached_packed_stream_type;
    typedef typename cached_packed_stream_type::iterator                                cached_packed_stream_iterator;
    typedef SparseStringSet<cached_packed_stream_iterator,LengthIterator>               cached_string_set;

    static cached_string_set make(
        const SparseStringSet<
            PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
            LengthIterator> string_set)
    {
        cached_packed_stream_type cached_packed_stream(
            cached_stream_iterator(
                uint4_iterator( cached_base_iterator( string_set.base_string().container().stream() ) )
                )
            );

        return cached_string_set(
            string_set.size(),
            cached_packed_stream.begin(),
            string_set.ranges() );
    }
};

template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
struct CachedPackedSparseStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    uint32>
{
    typedef const_cached_iterator<StreamIterator>                                       cached_stream_iterator;
    typedef PackedStream<cached_stream_iterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>  cached_packed_stream_type;
    typedef typename cached_packed_stream_type::iterator                                cached_packed_stream_iterator;
    typedef SparseStringSet<cached_packed_stream_iterator,LengthIterator>               cached_string_set;

    static cached_string_set make(
        const SparseStringSet<
            PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
            LengthIterator> string_set)
    {
        cached_packed_stream_type cached_packed_stream(
            cached_stream_iterator(
                    string_set.base_string().container().stream() )
            );

        return cached_string_set(
            string_set.size(),
            cached_packed_stream.begin(),
            string_set.ranges() );
    }
};

///
/// A utility function to convert a plain packed-sparse string set into a cached one
///
template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
typename CachedPackedSparseStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    typename std::iterator_traits<StreamIterator>::value_type>::cached_string_set
make_cached_string_set(
    const SparseStringSet<
        PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
        LengthIterator> string_set)
{
    typedef CachedPackedSparseStringSet<
        StreamIterator,
        SymbolType,
        SYMBOL_SIZE_T,
        BIG_ENDIAN_T,
        LengthIterator,
        typename std::iterator_traits<StreamIterator>::value_type> Adapter;

    return Adapter::make( string_set );
}

///
/// A utility function to convert a plain packed-concatenated string set into a cached one
///
template <
    typename StreamIterator,
    typename SymbolType,
    uint32   SYMBOL_SIZE_T,
    bool     BIG_ENDIAN_T,
    typename LengthIterator>
typename CachedPackedConcatStringSet<
    StreamIterator,
    SymbolType,
    SYMBOL_SIZE_T,
    BIG_ENDIAN_T,
    LengthIterator,
    typename std::iterator_traits<StreamIterator>::value_type>::cached_string_set
make_cached_string_set(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<StreamIterator,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
        LengthIterator> string_set)
{
    typedef CachedPackedConcatStringSet<
        StreamIterator,
        SymbolType,
        SYMBOL_SIZE_T,
        BIG_ENDIAN_T,
        LengthIterator,
        typename std::iterator_traits<StreamIterator>::value_type> Adapter;

    return Adapter::make( string_set );
}

} // namespace nvbio

#include <nvbio/basic/string_set_inl.h>
