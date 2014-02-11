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

namespace nvbio {


template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE ValueType fake_vector<ValueType,StorageType>::get(const index_type i) const
{
    return m_stream[i];
}
template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void fake_vector<ValueType,StorageType>::set(const index_type i, const ValueType v)
{
    #ifndef __CUDA_ARCH__
    if (ValueType(StorageType(v)) != v)
    {
        fprintf(stderr,"fake_vector: %lld out of storage-type range\n", int64(v));
        throw fake_vector_out_of_range();
    }
    #endif

    m_stream[i] = StorageType(v);
}

// return begin iterator
//
template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector<ValueType,StorageType>::iterator
fake_vector<ValueType,StorageType>::begin() const
{
    return iterator( m_stream, 0 );
}

/*
// dereference operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector_iterator<Stream>::Symbol
fake_vector_iterator<Stream>::operator* () const
{
    return m_stream.get( m_index );
}
*/
// dereference operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename fake_vector_iterator<Stream>::reference fake_vector_iterator<Stream>::operator* () const
{
    return reference( m_stream, m_index );
}

// indexing operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename fake_vector_iterator<Stream>::reference fake_vector_iterator<Stream>::operator[] (const index_type i) const
{
    return reference( m_stream, m_index + i );
}

// set value
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void fake_vector_iterator<Stream>::set(const value_type s)
{
    m_stream.set( m_index, s );
}

// pre-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator++ ()
{
    ++m_index;
    return *this;
}

// post-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator++ (int dummy)
{
    This r( m_stream, m_index );
    ++m_index;
    return r;
}

// pre-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator-- ()
{
    --m_index;
    return *this;
}

// post-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator-- (int dummy)
{
    This r( m_stream, m_index );
    --m_index;
    return r;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator+= (const index_type distance)
{
    m_index += distance;
    return *this;
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator-= (const index_type distance)
{
    m_index -= distance;
    return *this;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator+ (const index_type distance) const
{
    return This( m_stream, m_index + distance );
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator- (const index_type distance) const
{
    return This( m_stream, m_index - distance );
}

// difference
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector_iterator<Stream>::index_type
fake_vector_iterator<Stream>::operator- (const fake_vector_iterator it) const
{
    return index_type( m_index - it.m_index );
}

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index < it2.m_index;
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>(
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index > it2.m_index;
}

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator<= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index <= it2.m_index;
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>=(
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index >= it2.m_index;
}

// equality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return /*it1.m_stream == it2.m_stream &&*/ it1.m_index == it2.m_index;
}
// inequality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return /*it1.m_stream != it2.m_stream ||*/ it1.m_index != it2.m_index;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator= (const fake_vector_ref& ref)
{
    return (*this = value_type( ref ));
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator= (const value_type v)
{
    m_stream.set( m_index, v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator+= (const value_type v)
{
    m_stream.set( m_index, m_stream.get( m_index ) + v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator-= (const value_type v)
{
    m_stream.set( m_index, m_stream.get( m_index ) - v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator++ ()
{
    m_stream.set( m_index, m_stream.get( m_index )+1 );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator-- ()
{
    m_stream.set( m_index, m_stream.get( m_index )-1 );
    return *this;
}

// conversion operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>::operator value_type() const
{
    return m_stream.get( m_index );
}

} // namespace nvbio
