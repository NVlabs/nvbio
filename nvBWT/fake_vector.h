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
#include <iterator>

namespace nvbio {

struct fake_vector_out_of_range {};

///
/// fake_vector reference wrapper
///
template <typename Stream>
struct fake_vector_ref
{
    typedef typename Stream::value_type  value_type;
    typedef typename Stream::index_type  index_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref(Stream stream, index_type index)
        : m_stream( stream ), m_index( index ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref(const fake_vector_ref& ref)
        : m_stream( ref.m_stream ), m_index( ref.m_index ) {}

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator= (const fake_vector_ref& ref);

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator= (const value_type s);

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator+= (const value_type s);

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator-= (const value_type s);

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator++ ();

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref& operator-- ();

    /// conversion operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE operator value_type() const;

    Stream     m_stream;
    index_type m_index;
};

///
/// fake_vector iterator
///
template <typename Stream>
struct fake_vector_iterator
{
    typedef fake_vector_iterator<Stream> This;

    typedef typename Stream::value_type  value_type;
    typedef typename Stream::index_type  index_type;

    typedef fake_vector_ref<Stream>                           reference;
    typedef value_type                                        const_reference;
    typedef reference*                                        pointer;
    typedef index_type                                        difference_type;
    typedef typename std::random_access_iterator_tag          iterator_category;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator(Stream stream, const index_type index)
        : m_stream( stream ), m_index( index ) {}

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator* () const;

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) const;

    /// set value
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void set(const value_type s);

    /// pre-increment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator& operator++ ();

    /// post-increment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator operator++ (int dummy);

    /// pre-decrement operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator& operator-- ();

    /// post-decrement operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator operator-- (int dummy);

    /// add offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator& operator+= (const index_type distance);

    /// subtract offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator& operator-= (const index_type distance);

    /// add offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator operator+ (const index_type distance) const;

    /// subtract offset
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator operator- (const index_type distance) const;

    /// difference
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type operator- (const fake_vector_iterator it) const;

    Stream     m_stream;
    index_type m_index;
};

/// less than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

/// greater than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator> (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

/// less than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator<= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

/// greater than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

/// equality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

/// inequality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2);

///
/// A class to represent a packed stream of symbols, where the size of the
/// symbol is specified at compile-time as a template parameter.
///
template <typename ValueType, typename StorageType>
struct fake_vector
{
    typedef fake_vector<ValueType,StorageType> This;

    typedef ValueType                  value_type;
    typedef fake_vector_iterator<This> iterator;
    typedef fake_vector_ref<This>      reference;
    typedef int64                      index_type;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector() : m_stream(NULL) {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector(StorageType* stream) : m_stream( stream ) {}

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE value_type operator[] (const index_type i) const { return get(i); }
    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE reference operator[] (const index_type i) { return reference( *this, i ); }

    /// get the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE value_type get(const index_type i) const;

    /// set the i-th symbol
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void set(const index_type i, const value_type s);

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator begin() const;

private:
    StorageType* m_stream;
};

} // namespace nvbio

namespace std {

/// overload swap for fake_vector_ref to make sure it does the right thing
///
template <typename Stream>
void swap(
    nvbio::fake_vector_ref<Stream> ref1,
    nvbio::fake_vector_ref<Stream> ref2)
{
    typename nvbio::fake_vector_ref<Stream>::value_type tmp = ref1;

    ref1 = ref2;
    ref2 = tmp;
}

template <typename Stream>
void iter_swap(
    nvbio::fake_vector_iterator<Stream> it1,
    nvbio::fake_vector_iterator<Stream> it2)
{
    typename nvbio::fake_vector_iterator<Stream>::value_type tmp = *it1;

    it1.set( *it2 );
    it2.set( tmp );
}
template <typename Stream, typename OtherIterator>
void iter_swap(
    nvbio::fake_vector_iterator<Stream> it1,
    OtherIterator                       it2)
{
    typename nvbio::fake_vector_iterator<Stream>::value_type tmp = *it1;

    it1.set( *it2 );
    *it2 = tmp;
}
template <typename Stream, typename OtherIterator>
void iter_swap(
    OtherIterator                       it1,
    nvbio::fake_vector_iterator<Stream> it2)
{
    typename nvbio::fake_vector_iterator<Stream>::value_type tmp = *it2;

    it2.set( *it1 );
    *it1 = tmp;
}


} // std

#include "fake_vector_inl.h"
