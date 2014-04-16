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
#include <iterator>

namespace nvbio {

/// \page vector_wrappers_page Vector Wrappers
///
/// This module implements a vector adaptor, which allows to create an "std::vector"-like
/// container on top of a base iterator.
///
/// - vector_wrapper
///
/// \section VectorWrapperExampleSection Example
///
///\code
/// // build a vector_wrapper out of a static array
/// typedef vector_wrapper<uint32*> vector_type;
///
/// uint32 storage[16];
///
/// vector_type vector( 0, storage );
///
/// // use push_back()
/// vector.push_back( 3 );
/// vector.push_back( 7 );
/// vector.push_back( 11 );
///
/// // use resize()
/// vector.resize( 4 );
///
/// // use the indexing operator[]
/// vector[3] = 8;
///
/// // use the begin() / end() iterators
/// std::sort( vector.begin(), vector.end() );
///
/// // use front() and back()
/// printf("(%u, %u)\n");                       // -> (3,11)
///\endcode
///

///@addtogroup Basic
///@{

///
/// Wrapper class to create a "vector"-like container on top of a generic base iterator.
/// See \ref VectorWrapperExampleSection.
///
/// \tparam Iterator        base iterator type
///
template <typename Iterator>
struct vector_wrapper
{
    typedef Iterator                                                    iterator;
    typedef Iterator                                                    const_iterator;

    typedef typename std::iterator_traits<Iterator>::value_type         value_type;
    typedef typename std::iterator_traits<Iterator>::reference          reference;
    typedef typename to_const<reference>::type                          const_reference;
    typedef typename std::iterator_traits<Iterator>::pointer            pointer;
    typedef uint32                                                      size_type;
    typedef typename std::iterator_traits<Iterator>::difference_type    difference_type;
    //typedef typename std::iterator_traits<Iterator>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_wrapper() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_wrapper(const uint32 size, Iterator vec) : m_size( size ), m_vec( vec ) {}

    /// resize the vector
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void resize(const uint32 sz) { m_size = sz; }

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 length() const { return m_size; }

    /// return true iff size is null
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool empty() const { return m_size == 0; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_wrapper: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const uint32 i)             { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_wrapper: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

    /// push back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void push_back(const_reference val) { m_vec[ m_size ] = val; m_size++; }

    /// pop back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void pop_back() { --m_size; }

    /// return reference to front
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference front(void) const { return m_vec[0]; }

    /// return reference to front
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference front(void) { return m_vec[0]; }

    /// return reference to back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference back(void) const { return m_vec[m_size-1]; }

    /// return reference to back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference back(void) { return m_vec[m_size-1]; }

    /// return the base iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Iterator base() const { return m_vec; }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_iterator begin() const { return m_vec; }

    /// return end iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_iterator end() const { return m_vec + m_size; }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator begin() { return m_vec; }

    /// return end iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator end() { return m_vec + m_size; }

    uint32      m_size;
    Iterator    m_vec;
};

/// return length of a vector
///
template <typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 length(const vector_wrapper<Iterator>& vec) { return vec.length(); }

///@} Basic

} // namespace nvbio
