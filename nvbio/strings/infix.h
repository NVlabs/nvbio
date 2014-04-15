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

typedef uint32_2    string_infix_coord_type;
typedef uint64_2    long_string_infix_coord_type;

typedef uint32_4    string_set_infix_coord_type;
typedef uint64_4    long_string_set_infix_coord_type;

///@addtogroup Private
///@{

/// A class to represent a string infix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of infix coordinates, string_infix_coord_type for strings or string_set_infix_coord_type for string-sets
/// \tparam CoordDim            the number of coordinates, 2 for strings, 4 for string-sets
///
template <
    typename StringType,
    typename CoordType,
    uint32   CoordDim>
struct InfixCore {};

/// A class to represent a string infix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of infix coordinates, string_infix_coord_type for strings or string_set_infix_coord_type for string-sets
///
template <
    typename StringType,
    typename CoordType>
struct InfixCore<StringType,CoordType,2u>
{
    typedef StringType                                              string_type;
    typedef CoordType                                               coord_type;
    typedef typename vector_traits<CoordType>::value_type           index_type;

    typedef typename std::iterator_traits<string_type>::value_type  symbol_type;
    typedef typename std::iterator_traits<string_type>::value_type  value_type;
    typedef typename std::iterator_traits<string_type>::reference   reference;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixCore(
        const string_type   string,
        const coord_type    infix) :
        m_string( string ),
        m_coords( infix ) {}

    /// infix size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_coords.y - m_coords.x; }

    /// infix length
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

    /// return the infix range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 range() const { return make_uint2( m_coords.x, m_coords.y ); }

    /// return the infix coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type coords() const { return m_coords; }

    string_type     m_string;       ///< the underlying string set
    coord_type      m_coords;       ///< the infix coordinates
};

/// A class to represent a string infix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of infix coordinates, string_infix_coord_type for strings or string_set_infix_coord_type for string-sets
///
template <
    typename StringType,
    typename CoordType>
struct InfixCore<StringType,CoordType,4u>
{
    typedef StringType                                              string_type;
    typedef CoordType                                               coord_type;
    typedef typename vector_traits<CoordType>::value_type           index_type;

    typedef typename std::iterator_traits<string_type>::value_type  symbol_type;
    typedef typename std::iterator_traits<string_type>::value_type  value_type;
    typedef typename std::iterator_traits<string_type>::reference   reference;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixCore(
        const string_type   string,
        const coord_type    infix) :
        m_string( string ),
        m_coords( infix ) {}

    /// infix size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_coords.z - m_coords.y; }

    /// infix length
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 length() const { return size(); }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    symbol_type operator[] (const uint32 i) const { return m_string[ m_coords.y + i ]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[] (const uint32 i) { return m_string[ m_coords.y + i ]; }

    /// return the infix range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 range() const { return make_uint2( m_coords.y, m_coords.z ); }

    /// return the infix coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type coords() const { return m_coords; }

    /// return the string id
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 string_id() const { return m_coords.x; }

    string_type     m_string;       ///< the underlying string set
    coord_type      m_coords;       ///< the infix coordinates
};

///@} Private

/// A class to represent a string infix, i.e. an arbitrarily placed substring
///
/// \tparam StringType          the underlying string type
/// \tparam CoordType           the type of infix coordinates, string_infix_coord_type for strings or string_set_infix_coord_type for string-sets
/// \tparam CoordDim            the number of coordinates, 2 for strings, 4 for string-sets
///
template <
    typename StringType,
    typename CoordType>
struct Infix : public InfixCore< StringType, CoordType, vector_traits<CoordType>::DIM >
{
    typedef InfixCore< StringType, CoordType, vector_traits<CoordType>::DIM >   core_type;
    typedef StringType                                                          string_type;
    typedef CoordType                                                           coord_type;
    typedef typename vector_traits<CoordType>::value_type                       index_type;

    typedef typename std::iterator_traits<string_type>::value_type              symbol_type;
    typedef typename std::iterator_traits<string_type>::value_type              value_type;
    typedef typename std::iterator_traits<string_type>::reference               reference;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Infix() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Infix(
        const string_type   string,
        const coord_type    infix) : core_type( string, infix ) {}
};

/// make an infix, i.e. a substring of a given string
///
/// \tparam StringType  the underlying string type
/// \tparam CoordType   the coordinates type, either string_infix_coord_type or string_set_infix_coord_type
///
/// \param string       the underlying string object
/// \param coords       the infix coordinates
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Infix<StringType,CoordType> make_infix(const StringType string, const CoordType coords)
{
    return Infix<StringType,CoordType>( string, coords );
}

/// Represent a set of infixes of a string or string-set
///
/// \tparam SequenceType            the string or string-set type
/// \tparam InfixIterator           the infix iterator type - value_type can be string_infix_coord_type for strings, string_set_infix_coord_type for string-sets
/// \tparam CoordDim                the number of coordinates representing an infix, 2 for strings, 4 for string-sets
///
template <
    typename SequenceType,
    typename InfixIterator,
    uint32   CoordDim>
struct InfixSetCore {};

/// Represent a set of infixes of a string
///
/// \tparam SequenceType            the string or string-set container
/// \tparam InfixIterator           the infix iterator type - value_type can be string_infix_coord_type or long_string_set_infix_coord_type
///
template <
    typename SequenceType,
    typename InfixIterator>
struct InfixSetCore<SequenceType,InfixIterator,2u>
{
    typedef SequenceType                                                sequence_type;
    typedef InfixIterator                                               infix_iterator;

    typedef typename std::iterator_traits<InfixIterator>::value_type    coord_type;
    typedef Infix<sequence_type, coord_type>                            string_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSetCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSetCore(
        const uint32            size,
        const sequence_type     sequence,
        const infix_iterator    infixes) :
        m_size( size ),
        m_sequence( sequence ),
        m_infixes( infixes ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const coord_type coords = m_infixes[i];
        return string_type( m_sequence, coords );
    }

    uint32              m_size;
    sequence_type       m_sequence;
    infix_iterator      m_infixes;
};

/// Represent a set of infixes of a string-set
///
/// \tparam SequenceType            the string or string-set type
/// \tparam InfixIterator           the infix iterator type - value_type can be uint4 or uint64_4
///
template <
    typename SequenceType,
    typename InfixIterator>
struct InfixSetCore<SequenceType,InfixIterator,4u>
{
    typedef SequenceType                                                sequence_type;
    typedef InfixIterator                                               infix_iterator;

    typedef typename sequence_type::string_type                         base_string_type;
    typedef typename std::iterator_traits<InfixIterator>::value_type    coord_type;
    typedef Infix<base_string_type, coord_type>                         string_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSetCore() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSetCore(
        const uint32            size,
        const sequence_type     sequence,
        const infix_iterator    infixes) :
        m_size( size ),
        m_sequence( sequence ),
        m_infixes( infixes ) {}

    /// set size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// indexing operator: access the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type operator[] (const uint32 i) const
    {
        const coord_type coords = m_infixes[i];
        return string_type( m_sequence[ coords.x ], coords );
    }

    uint32              m_size;
    sequence_type       m_sequence;
    infix_iterator      m_infixes;
};

///@} Private

/// return the string begin of a given infix
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename Infix<StringType,CoordType>::index_type infix_begin(const Infix<StringType,CoordType>& infix) { return infix.range().x; }

/// return the string end of a given infix
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename Infix<StringType,CoordType>::index_type infix_end(const Infix<StringType,CoordType>& infix) { return infix.range().y; }

/// return the string index of a given infix
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 string_id(const InfixCore<StringType,CoordType,4u>& infix) { return infix.string_id(); }

/// return the length a given infix
///
template <typename StringType, typename CoordType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 length(const Infix<StringType,CoordType>& infix) { return infix.length(); }

/// Represent a set of infixes of a string or string-set. An InfixSet is a \ref StringSetAnchor "String Set".
///
/// \tparam SequenceType        the string or string-set type
/// \tparam InfixIterator       the infix iterator type - value_type can be string_infix_coord_type for strings, string_set_infix_coord_type for string-sets
///
template <
    typename SequenceType,
    typename InfixIterator>
struct InfixSet : public InfixSetCore<
    SequenceType,
    InfixIterator,
    vector_traits<typename std::iterator_traits<InfixIterator>::value_type>::DIM>
{
    typedef InfixSetCore<
        SequenceType,
        InfixIterator,
        vector_traits<typename std::iterator_traits<InfixIterator>::value_type>::DIM>   base_type;

    typedef SequenceType                                                sequence_type;  ///< the underlying sequence type
    typedef InfixIterator                                               infix_iterator; ///< the underlingy infix iterator type

    typedef typename base_type::coord_type                              coord_type;     ///< the infix coordinates type
    typedef typename base_type::string_type                             string_type;    ///< the infix string type

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSet() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InfixSet(
        const uint32            size,
        const sequence_type     sequence,
        const infix_iterator    infixes) :
        base_type( size, sequence, infixes ) {}
};

///@} StringSetsModule
///@} Strings

} // namespace nvbio

//#include <nvbio/basic/infix_inl.h>
