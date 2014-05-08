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

#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/vector_view.h>

#pragma once

namespace nvbio {

///@addtogroup IO
///@{

///@addtogroup Utils
///@{

enum ReadType { STANDARD = 0u, COMPLEMENT = 1u };
enum DirType  { FORWARD  = 0u, REVERSE    = 1u };

///
/// A utility functor to reverse a given stream
///
template<typename IndexType>
struct ReverseXform
{
    typedef IndexType index_type;
    const index_type pos;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ReverseXform() : pos(0) { }

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ReverseXform(const index_type n) : pos(n-1) { }

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_type operator() (const index_type i) const { return pos-i; }
};

///
/// A utility functor to apply an offset to a given a stream
///
template<typename IndexType>
struct OffsetXform
{
    typedef IndexType index_type;
    const index_type pos;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    OffsetXform() : pos(0) { }

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    OffsetXform(const index_type n) : pos(n) { }

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_type operator() (const index_type i) const { return pos+i; }
};

struct quality_nop {};

///
/// Helper class to extract qualities from a read
///
template <typename ReadStreamType>
struct ReadStreamQualities
{
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStreamQualities() : m_read( NULL ) {}

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStreamQualities(const ReadStreamType& read) : m_read( &read ) {}

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint8 operator[] (const uint32 pos) const { return m_read->quality(pos); }

    const ReadStreamType* m_read;
};

///
/// Helper class to represent a read or its reverse-complement - useful to avoid warp-divergence.
///
template< typename StreamType, typename QualType = quality_nop >
struct ReadStream
{
    typedef typename StreamType::symbol_type value_type;
    typedef ReadStream<StreamType,QualType>  this_type;
    typedef ReadStreamQualities<this_type>   qual_string_type; 

    /// default constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream() {}

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream(const StreamType& s, const uint2 range) 
      : stream(s), first(range.x), last(range.y-1), rev(false), comp(false) { }

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream(const StreamType& s, const QualType q, const uint2 range) 
      : stream(s), qual(q), first(range.x), last(range.y-1), rev(false), comp(false) { }

    /// set the read flags
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    void set_flags(const DirType d, const ReadType t) { rev = (d == REVERSE); comp = (t == COMPLEMENT); }

    /// return string length
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint32 length() const { return 1+last-first; }

    /// return a given base
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    value_type operator[] (uint32 pos) const
    {
        const uint32 index = rev ? last-pos : first+pos;
        const value_type c = stream[ index ];

        return (comp) ? (c < 4 ? 3-c : c) : c;
    }

    /// return a given base quality
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint8 quality(const uint32 pos) const
    {
        const uint32 index = rev ? last-pos : first+pos;
        return qual[ index ];
    }

    /// return qualities
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    qual_string_type qualities() const { return qual_string_type(*this); }

    bool        rev;                ///< reverse flag
    bool        comp;               ///< complement flag
    uint32      first, last;        ///< offset of first and last elements
    StreamType  stream;             ///< read stream
    QualType    qual;               ///< quality stream
};

/// return string length
///
template< typename StreamType, typename QualType>
NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
uint32 length(const ReadStream<StreamType,QualType>& read) { return read.length(); }

///
/// Utility class to load a read with a StringLoader.
/// The Tag type allows to specify the caching policy.
///
template <typename BatchType, typename Tag>
struct ReadLoader
{
    typedef typename BatchType::read_storage_iterator                                              read_storage;
    typedef typename BatchType::qual_storage_iterator                                              qual_iterator;
    typedef PackedStringLoader<read_storage, io::ReadData::READ_BITS, io::ReadData::HI_BITS,Tag>   loader_type;
    typedef typename loader_type::iterator                                                         read_iterator;
    typedef ReadStream<read_iterator,qual_iterator>                                                string_type;

    /// load a read
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const BatchType& batch, const uint2 range, const DirType dir, const ReadType op)
    {
        const qual_iterator quals( batch.qual_stream() + range.x );

        string_type read(
            loader.load( batch.read_stream_storage(), range.x, range.y - range.x ),
            quals,
            make_uint2( 0, range.y - range.x ) );

        read.set_flags( dir, op );
        return read;
    }
    /// load a read substring
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const BatchType& batch, const uint2 range, const DirType dir, const ReadType op, const uint2 subrange)
    {
        const qual_iterator quals( batch.qual_stream() + range.x );

        string_type read(
            loader.load( batch.read_stream_storage(), range.x, range.y - range.x, subrange, dir == REVERSE ? true : false ),
            quals,
            make_uint2( 0, range.y - range.x ) );

        read.set_flags( dir, op );
        return read;
    }

    loader_type loader;
};

///
/// Utility class to load a genome substring with a StringLoader.
/// The Tag type allows to specify the caching policy.
///
template <typename GenomeStorage, typename Tag>
struct GenomeLoader
{
    typedef PackedStringLoader<GenomeStorage,2,true,Tag>    loader_type;
    typedef typename loader_type::iterator                  iterator;
    typedef vector_view<iterator>                           string_type;

    /// load a genome segment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const GenomeStorage ptr, const uint32 offset, const uint32 len)
    {
        return string_type( len, loader.load( ptr, offset, len ) );
    }
    /// load a substring of a genome segment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const GenomeStorage ptr, const uint32 offset, const uint32 len, const uint2 subrange)
    {
        return string_type( len, loader.load( ptr, offset, len, subrange, STANDARD ) );
    }

    loader_type loader;
};

///@} // Utils
///@} // IO

} // namespace nvbio
