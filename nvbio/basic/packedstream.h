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
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/strided_iterator.h>
#include <nvbio/basic/iterator.h>
#if defined(__CUDACC__)
#include <nvbio/basic/cuda/arch.h>
#endif

#if defined(BIG_ENDIAN)
#undef BIG_ENDIAN
#endif

namespace nvbio {

/// \page packed_streams_page Packed Streams
///
/// This module implements interfaces to hold binary packed streams expressed using compile-time specified alphabet sizes.
/// The idea is that a packed stream is an open-ended sequence of symbols encoded with a fixed number of bits on an underlying
/// stream of words.
/// The words themselves can be of different types, ranging from uint32 to uint4, to support different kind of memory access
/// patterns.
///
/// \section AtAGlanceSection At a Glance
///
/// The main classes are:
///
/// - PackedVector :             a packed vector object
/// - PackedStream :             a packed stream object
/// - PackedStreamRef :          a proxy object to represent packed symbol references
/// - PackedStreamIterator :     a PackedStream iterator
/// - \ref PackedStringLoaders : a packed stream loader which allows to cache portions of a packed stream into
///                              different memory spaces (e.g. local memory)
/// - StreamTransform :          a helper class to remap accesses to a stream through a simple transformation
///
/// \section ExampleSection Example
///
///\code
/// // pack 16 DNA symbols using a 2-bit alphabet into a single word
/// uint32 word;
/// PackedStream<uint32,uint8,2u,false> packed_string( &word );
///
/// const uint32 string_len = 16;
/// const char   string[]   = "ACGTTGCAACGTTGCA";
/// for (uint32 i = 0; i < string_len; ++i)
///     packed_string[i] = char_to_dna( string[i] );
///
/// // and count the occurrences of T
/// const uint32 occ = util::count_occurrences( packed_string.begin(), string_len, char_to_dna('T') );
///\endcode
///
/// \section TechnicalDetailsSection Technical Details
///
/// A detailed description of all the classes and functions in this module can be found in the
/// \ref PackedStreams module documentation.
///

///@addtogroup Basic
///@{

///@defgroup PackedStreams Packed Streams
/// This module implements interfaces to hold binary packed streams expressed using compile-time specified alphabet sizes.
/// The idea is that a packed stream is an open-ended sequence of symbols encoded with a fixed number of bits on an underlying
/// stream of words.
/// The words themselves can be of different types, ranging from uint32 to uint4, to support different kind of memory access
/// patterns.
///@{

/// Basic stream traits class, providing compile-time information about a string type
///
template <typename T> struct stream_traits
{
    typedef uint32 index_type;
    typedef char   symbol_type;

    static const uint32 SYMBOL_SIZE  = 8u;
    static const uint32 SYMBOL_COUNT = 256u;
};

/// T* specialization of the stream_traits class, providing compile-time information about a string type
///
template <typename T> struct stream_traits<T*>
{
    typedef uint32 index_type;
    typedef T      symbol_type;

    static const uint32 SYMBOL_SIZE  = uint32( 8u * sizeof(T) );
    static const uint32 SYMBOL_COUNT = uint32( (uint64(1u) << SYMBOL_SIZE) - 1u );
};

/// const T* specialization of the stream_traits class, providing compile-time information about a string type
///
template <typename T> struct stream_traits<const T*>
{
    typedef uint32 index_type;
    typedef T      symbol_type;

    static const uint32 SYMBOL_SIZE  = uint32( 8u * sizeof(T) );
    static const uint32 SYMBOL_COUNT = uint32( (uint64(1u) << SYMBOL_SIZE) - 1u );
};

///
/// PackedStream reference wrapper
///
template <typename Stream>
struct PackedStreamRef
{
    typedef typename Stream::symbol_type Symbol;
    typedef typename Stream::symbol_type symbol_type;
    typedef symbol_type                  value_type;
    typedef typename Stream::index_type  index_type;
    typedef typename Stream::sindex_type sindex_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamRef(Stream stream, index_type index)
        : m_stream( stream ), m_index( index ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamRef(const PackedStreamRef& ref)
        : m_stream( ref.m_stream ), m_index( ref.m_index ) {}

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamRef& operator= (const PackedStreamRef& ref);

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamRef& operator= (const Symbol s);

    /// conversion operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE operator Symbol() const;

    Stream     m_stream;
    index_type m_index;
};

/// redefine the to_const meta-function for PackedStreamRef to just return a symbol
///
template <typename Stream> struct to_const< PackedStreamRef<Stream> >
{
    typedef typename PackedStreamRef<Stream>::symbol_type type;
};

///
/// PackedStream iterator
///
template <typename Stream>
struct PackedStreamIterator
{
    typedef PackedStreamIterator<Stream> This;

    static const uint32 SYMBOL_SIZE  = stream_traits<Stream>::SYMBOL_SIZE;
    static const uint32 SYMBOL_COUNT = stream_traits<Stream>::SYMBOL_COUNT;

    typedef typename Stream::symbol_type Symbol;
    typedef typename Stream::symbol_type symbol_type;
    typedef typename Stream::index_type  index_type;
    typedef typename Stream::sindex_type sindex_type;

    typedef Stream                                                      stream_type;
    typedef Symbol                                                      value_type;
    typedef PackedStreamRef<Stream>                                     reference;
    typedef Symbol                                                      const_reference;
    typedef reference*                                                  pointer;
    typedef int32                                                       difference_type;
    typedef int32                                                       distance_type;
    typedef typename std::iterator_traits<Stream>::iterator_category    iterator_category;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator(Stream stream, const index_type index)
        : m_stream( stream ), m_index( index ) {}

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator* () const;

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const sindex_type i) const;

    /// indexing operator
    ///
    //NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) const;

    /// set value
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void set(const Symbol s);

    /// pre-increment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator& operator++ ();

    /// post-increment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator operator++ (int dummy);

    /// pre-decrement operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator& operator-- ();

    /// post-decrement operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator operator-- (int dummy);

    /// add offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator& operator+= (const sindex_type distance);

    /// subtract offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator& operator-= (const sindex_type distance);

    /// add offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator operator+ (const sindex_type distance) const;

    /// subtract offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStreamIterator operator- (const sindex_type distance) const;

    /// difference
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE sindex_type operator- (const PackedStreamIterator it) const;

    /// return the container this iterator refers to
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Stream container() const { return m_stream; }

    /// return the offset this iterator refers to
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_type index() const { return m_index; }

    Stream     m_stream;
    index_type m_index;
};

/// PackedStreamIterator<Stream> specialization of the stream_traits class, providing compile-time information about
/// the corresponding string type
///
template <typename Stream>
struct stream_traits< PackedStreamIterator<Stream> >
{
    typedef typename PackedStreamIterator<Stream>::index_type   index_type;
    typedef typename PackedStreamIterator<Stream>::symbol_type  symbol_type;

    static const uint32 SYMBOL_SIZE  = PackedStreamIterator<Stream>::SYMBOL_SIZE;
    static const uint32 SYMBOL_COUNT = PackedStreamIterator<Stream>::SYMBOL_COUNT;
};

/// less than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const PackedStreamIterator<Stream>& it1,
    const PackedStreamIterator<Stream>& it2);

/// greater than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator> (
    const PackedStreamIterator<Stream>& it1,
    const PackedStreamIterator<Stream>& it2);

/// equality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const PackedStreamIterator<Stream>& it1,
    const PackedStreamIterator<Stream>& it2);

/// inequality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const PackedStreamIterator<Stream>& it1,
    const PackedStreamIterator<Stream>& it2);

///
/// A class to represent a packed stream of symbols, where the size of the
/// symbol is specified at compile-time as a template parameter.
/// The sequence is packed on top of an underlying stream of words, whose type can also be specified at compile-time
/// in order to allow for different memory access patterns.
///
/// \tparam InputStream         the underlying stream of words used to hold the packed stream (e.g. uint32, uint4)
/// \tparam Symbol              the unpacked symbol type (e.g. uint8)
/// \tparam SYMBOL_SIZE_T       the number of bits needed for each symbol
/// \tparam BIG_ENDIAN_T        the "endianness" of the words: if true, symbols will be packed from right to left within each word
/// \tparam IndexType           the type of integer used to address the stream (e.g. uint32, uint64)
///
template <typename InputStream, typename Symbol, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType = uint32>
struct PackedStream
{
    typedef PackedStream<InputStream,Symbol,SYMBOL_SIZE_T, BIG_ENDIAN_T,IndexType> This;

    static const uint32 SYMBOL_SIZE  = SYMBOL_SIZE_T;
    static const uint32 SYMBOL_COUNT = 1u << SYMBOL_SIZE;
    static const uint32 SYMBOL_MASK  = SYMBOL_COUNT - 1u;
    static const uint32 BIG_ENDIAN   = BIG_ENDIAN_T;
    static const uint32 ALPHABET_SIZE = SYMBOL_COUNT;

    typedef typename unsigned_type<IndexType>::type                  index_type;
    typedef typename   signed_type<IndexType>::type                 sindex_type;
    typedef typename std::iterator_traits<InputStream>::value_type  storage_type;

    typedef InputStream                                             stream_type;
    typedef Symbol                                                  symbol_type;
    typedef PackedStreamIterator<This>                              iterator;
    typedef PackedStreamRef<This>                                   reference;
    typedef reference*                                              pointer;
    typedef typename std::random_access_iterator_tag                iterator_category;
    typedef symbol_type                                             value_type;
    typedef sindex_type                                             difference_type;
    typedef sindex_type                                             distance_type;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStream() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE PackedStream(const InputStream stream) : m_stream( stream ) {}

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Symbol operator[] (const index_type i) const { return get(i); }
    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) { return reference( *this, i ); }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Symbol get(const index_type i) const;

    /// set the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void set(const index_type i, const Symbol s);

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator begin() const;

    /// return the base stream
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InputStream stream() const { return m_stream; }

private:
    InputStream m_stream;
};

/// assign a sequence to a packed stream
///
template <typename InputIterator, typename InputStream, typename Symbol, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
NVBIO_HOST_DEVICE
void assign(
    const uint32                                                                                    input_len,
    InputIterator                                                                                   input_string,
    PackedStreamIterator< PackedStream<InputStream,Symbol,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType> >   packed_string);

/// PackedStream specialization of the stream_traits class, providing compile-time information about the
/// corresponding string type
///
template <typename InputStream, typename SymbolType, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
struct stream_traits< PackedStream<InputStream,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType> >
{
    typedef IndexType   index_type;
    typedef SymbolType  symbol_type;

    static const uint32 SYMBOL_SIZE  = PackedStream<InputStream,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::SYMBOL_SIZE;
    static const uint32 SYMBOL_COUNT = PackedStream<InputStream,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::SYMBOL_COUNT;
};

///
/// A simple class to remap accesses to a stream through an index
/// transformation.
///
template <typename InputStream, typename IndexTransform>
struct StreamRemapper
{
    static const uint32 SYMBOL_SIZE  = stream_traits<InputStream>::SYMBOL_SIZE;
    static const uint32 SYMBOL_COUNT = stream_traits<InputStream>::SYMBOL_COUNT;

    typedef StreamRemapper<InputStream,IndexTransform> This;
    typedef typename InputStream::symbol_type    symbol_type;
    typedef PackedStreamIterator<This>           iterator;
    typedef PackedStreamRef<This>                reference;
    typedef typename InputStream::index_type     index_type;
    typedef typename InputStream::sindex_type    sindex_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StreamRemapper() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StreamRemapper(InputStream stream, const IndexTransform transform)
        : m_stream( stream ), m_transform( transform ) {}

    /// set
    ///
    /// \param i        requested value
    /// \param s        symbol
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void set(const index_type i, const symbol_type s)
    {
        return m_stream.set( m_transform(i), s );
    }
    /// get
    ///
    /// \param i        requested value
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE symbol_type get(const index_type i) const
    {
        return m_stream.get( m_transform(i) );
    }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE symbol_type operator[] (const index_type i) const { return get(i); }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) { return reference( *this, i ); }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator begin() const
    {
        return iterator( *this, 0u );
    }

private:
    InputStream     m_stream;
    IndexTransform  m_transform;
};

///
/// A simple class to remap accesses to a stream through an index
/// transformation.
///
template <typename InputStream, typename Transform>
struct StreamTransform
{
    static const uint32 SYMBOL_SIZE  = stream_traits<InputStream>::SYMBOL_SIZE;
    static const uint32 SYMBOL_COUNT = stream_traits<InputStream>::SYMBOL_COUNT;

    typedef StreamTransform<InputStream,Transform> This;
    typedef typename InputStream::symbol_type    symbol_type;
    typedef PackedStreamIterator<This>           iterator;
    typedef PackedStreamRef<This>                reference;
    typedef typename InputStream::index_type     index_type;
    typedef typename InputStream::sindex_type    sindex_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StreamTransform() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    StreamTransform(InputStream stream, const Transform transform)
        : m_stream( stream ), m_transform( transform ) {}

    /// get
    ///
    /// \param i        requested value
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE symbol_type get(const index_type i) const
    {
        return m_transform( m_stream.get(i) );
    }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE symbol_type operator[] (const index_type i) const { return get(i); }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) { return reference( *this, i ); }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator begin() const
    {
        return iterator( *this, 0u );
    }

private:
    InputStream m_stream;
    Transform   m_transform;
};

///
/// A utility class to view a uint4 iterator as a uint32 one
///
template <typename IteratorType>
struct uint4_as_uint32_iterator
{
    typedef uint32                                                          value_type;
    typedef value_type*                                                     pointer;
    typedef value_type                                                      reference;
    typedef typename std::iterator_traits<IteratorType>::difference_type    difference_type;
    //typedef typename std::iterator_traits<IteratorType>::distance_type      distance_type;
    typedef typename std::iterator_traits<IteratorType>::iterator_category  iterator_category;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint4_as_uint32_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint4_as_uint32_iterator(const IteratorType it) : m_it( it )  {}

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator[] (const uint32 i) const
    {
        const uint32 c = i & 3u;
        const uint32 w = i >> 2;
        return nvbio::comp( m_it[w], c );
    }

    IteratorType m_it;
};

///
/// A utility device function to transpose a set of packed input streams:
///   the symbols of the i-th input stream is supposed to be stored contiguously in the range [offset(i), offset + N(i)]
///   the *words* of i-th output stream will be stored in strided fashion at out_stream[tid, tid + (N(i)+symbols_per_word-1/symbols_per_word) * stride]
///
/// The function is warp-synchronous, hence all threads in each warp must be active.
///
/// \param stride       output stride
/// \param N            length of this thread's string in the input stream
/// \param in_offset    offset of this thread's string in the input stream
/// \param in_stream    input stream
/// \param out_stream   output stream (usually of the form ptr + thread_id)
///
template <uint32 BLOCKDIM, uint32 BITS, bool BIG_ENDIAN, typename InStreamIterator, typename OutStreamIterator>
NVBIO_HOST_DEVICE
void transpose_packed_streams(const uint32 stride, const uint32 N, const uint32 in_offset, const InStreamIterator in_stream, OutStreamIterator out_stream);

///@} PackedStreams
///@} Basic

} // namespace nvbio

namespace std {

/// overload swap for PackedStreamRef to make sure it does the right thing
///
template <typename Stream>
void swap(
    nvbio::PackedStreamRef<Stream> ref1,
    nvbio::PackedStreamRef<Stream> ref2)
{
    typename nvbio::PackedStreamRef<Stream>::value_type tmp = ref1;

    ref1 = ref2;
    ref2 = tmp;
}

template <typename Stream>
void iter_swap(
    nvbio::PackedStreamIterator<Stream> it1,
    nvbio::PackedStreamIterator<Stream> it2)
{
    typename nvbio::PackedStreamIterator<Stream>::value_type tmp = *it1;

    it1.set( *it2 );
    it2.set( tmp );
}

} // std

#include <nvbio/basic/packedstream_inl.h>
