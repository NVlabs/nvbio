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

#include <nvbio/strings/string_set.h>


namespace nvbio {

///@addtogroup Strings
///@{

///@addtogroup StringSetsModule
///@{

/// A class to represent a string prefix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of prefix coordinates, uint32 or uint64 for strings, uint2 or uint64_2 for string-sets
/// \tparam CoordDim            the number of coordinates, 1 for strings, 2 for string-sets
///
template <
    typename StringType,
    typename CoordType,
    uint32   CoordDim>
struct Prefix {};

///@addtogroup Private
///@{

/// A class to represent a string prefix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of prefix coordinates, uint32|uint64
///
template <
    typename StringType,
    typename CoordType>
struct Prefix<StringType,CoordType,1u>
{
    typedef StringType                                              string_type;
    typedef CoordType                                               coord_type;

    typedef typename std::iterator_traits<string_type>::value_type  symbol_type;
    typedef typename std::iterator_traits<string_type>::value_type  value_type;
    typedef typename std::iterator_traits<string_type>::reference   reference;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Prefix() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Prefix(
        const string_type   string,
        const coord_type    prefix) :
        m_string( string ),
        m_coords( prefix ) {}

    /// prefix size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return nvbio::length( m_string ) - m_coords; }

    /// prefix length
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 length() const { return size(); }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_type operator[] (const uint32 i) const { return m_string[ m_coords + i ]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[] (const uint32 i) { return m_string[ m_coords + i ]; }

    /// return the prefix coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type coords() const { return m_coords; }

    string_type     m_string;       ///< the underlying string set
    coord_type      m_coords;       ///< the prefix coordinates
};

/// A class to represent a string prefix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of prefix coordinates, uint32|uint64
///
template <
    typename StringType,
    typename CoordType>
struct Prefix<StringType,CoordType,2u>
{
    typedef StringType                                              string_type;
    typedef CoordType                                               coord_type;

    typedef typename std::iterator_traits<string_type>::value_type  symbol_type;
    typedef typename std::iterator_traits<string_type>::value_type  value_type;
    typedef typename std::iterator_traits<string_type>::reference   reference;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Prefix() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Prefix(
        const string_type   string,
        const coord_type    prefix) :
        m_string( string ),
        m_coords( prefix ) {}

    /// prefix size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return nvbio::length( m_string ) - m_coords.x; }

    /// prefix length
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 length() const { return size(); }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_type operator[] (const uint32 i) const { return m_string[ m_coords.x + i ]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[] (const uint32 i) { return m_string[ m_coords.x + i ]; }

    /// return the prefix coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type coords() const { return m_coords; }

    string_type     m_string;       ///< the underlying string set
    coord_type      m_coords;       ///< the prefix coordinates
};

/// Represent a set of prefixes of a string or string-set
///
/// \tparam SequenceType            the string or string-set type
/// \tparam PrefixIterator          the prefix iterator type - value_type can be uint32 or uint64 for strings, uint2 or uint64_2 for string-sets
/// \tparam CoordDim                the number of coordinates representing a prefix, 1 for strings, 2 for string-sets
///
template <
    typename SequenceType,
    typename PrefixIterator,
    uint32   CoordDim>
struct PrefixSetCore {};

/// Represent a set of prefixes of a string
///
/// \tparam SequenceType            the string or string-set container
/// \tparam PrefixIterator          the prefix iterator type - value_type can be uint32 or uint64
///
template <
    typename SequenceType,
    typename PrefixIterator>
struct PrefixSetCore<SequenceType,PrefixIterator,1u>
{
    typedef SequenceType                                                sequence_type;
    typedef PrefixIterator                                              prefix_iterator;

    typedef typename std::iterator_traits<PrefixIterator>::value_type   coord_type;
    typedef Prefix<sequence_type, coord_type, 1u>                       string_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSetCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSetCore(
        const uint32            size,
        const sequence_type     sequence,
        const prefix_iterator    prefixes) :
        m_size( size ),
        m_sequence( sequence ),
        m_prefixes( prefixes ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const coord_type coords = m_prefixes[i];
        return string_type( m_sequence, coords );
    }

    uint32              m_size;
    sequence_type       m_sequence;
    prefix_iterator     m_prefixes;
};

/// Represent a set of prefixes of a string-set
///
/// \tparam SequenceType            the string or string-set type
/// \tparam PrefixIterator          the prefix iterator type - value_type can be uint2 or uint64_2
///
template <
    typename SequenceType,
    typename PrefixIterator>
struct PrefixSetCore<SequenceType,PrefixIterator,2u>
{
    typedef SequenceType                                                sequence_type;
    typedef PrefixIterator                                              prefix_iterator;

    typedef typename sequence_type::string_type                         base_string_type;
    typedef typename std::iterator_traits<PrefixIterator>::value_type   coord_type;
    typedef Prefix<base_string_type, coord_type, 2u>                    string_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSetCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSetCore(
        const uint32            size,
        const sequence_type     sequence,
        const prefix_iterator   prefixes) :
        m_size( size ),
        m_sequence( sequence ),
        m_prefixes( prefixes ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const coord_type coords = m_prefixes[i];
        return string_type( m_sequence[ coords.y ], coords );
    }

    uint32              m_size;
    sequence_type       m_sequence;
    prefix_iterator     m_prefixes;
};

///@} Private

/// return the string index of a given prefix
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 string_id(const Prefix<StringType,CoordType,2u>& prefix) { return prefix.m_coords.y; }

/// return the length of a given prefix
///
template <typename StringType, typename CoordType, uint32 CoordDim>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 length(const Prefix<StringType,CoordType,CoordDim>& prefix) { return prefix.length(); }

/// Represent a set of prefixes of a string or string-set. An PrefixSet is a \ref StringSetAnchor "String Set".
///
/// \tparam SequenceType        the string or string-set type
/// \tparam PrefixIterator      the prefix iterator type - value_type can be uint32 or uint64 for strings, uint2 or uint64_2 for string-sets
///
template <
    typename SequenceType,
    typename PrefixIterator>
struct PrefixSet : public PrefixSetCore<
                            SequenceType,
                            PrefixIterator,
                            vector_traits<typename std::iterator_traits<PrefixIterator>::value_type>::DIM>
{
    typedef PrefixSetCore<
        SequenceType,
        PrefixIterator,
        vector_traits<typename std::iterator_traits<PrefixIterator>::value_type>::DIM>   base_type;

    typedef SequenceType                                                sequence_type;      ///< the underlying sequence type
    typedef PrefixIterator                                              prefix_iterator;    ///< the underlingy prefix iterator type

    typedef typename base_type::coord_type                              coord_type;         ///< the prefix coordinates type
    typedef typename base_type::string_type                             string_type;        ///< the prefix string type

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSet() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    PrefixSet(
        const uint32            size,
        const sequence_type     sequence,
        const prefix_iterator    prefixes) :
        base_type( size, sequence, prefixes ) {}
};

///@} StringSetsModule
///@} Strings

} // namespace nvbio

//#include <nvbio/basic/prefix_inl.h>
