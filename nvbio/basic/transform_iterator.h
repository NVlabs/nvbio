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
#include <nvbio/basic/iterator.h>

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// Wrapper class to create a transform iterator out of another base iterator
/// and a function
///
template <typename T, typename Transform>
struct transform_iterator
{
    typedef typename Transform::result_type                     value_type;
    typedef typename value_type&                                reference;
    typedef typename value_type&                                const_reference;
    typedef typename value_type*                                pointer;
    typedef typename std::iterator_traits<T>::difference_type   difference_type;
    //typedef typename std::iterator_traits<T>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                     iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator(const T base, const Transform f) : m_base( base ), m_f( f ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator(const transform_iterator& it) : m_base( it.m_base ), m_f( it.m_f ) {}

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { return m_f( m_base[i] ); }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator*() const { return m_f( m_base[0] ); }

    /// pre-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator++(int i)
    {
        transform_iterator<T,Transform> r( m_base, m_f );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator--(int i)
    {
        transform_iterator<T,Transform> r( m_base, m_f );
        --m_base;
        return r;
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator+(const difference_type i) const
    {
        return transform_iterator( m_base + i, m_f );
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator-(const difference_type i) const
    {
        return transform_iterator( m_base - i, m_f );
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const transform_iterator<T,Transform> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator& operator=(const transform_iterator<T,Transform>& it)
    {
        m_base = it.m_base;
        m_f    = it.m_f;
        return *this;
    }

    T         m_base;
    Transform m_f;
};

/// make a transform_iterator
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
transform_iterator<T,Transform> make_transform_iterator(const T it, const Transform f)
{
    return transform_iterator<T,Transform>( it, f );
}


/// operator ==
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator==(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2)
{
    return (it1.m_base == it2.m_base) && (it1.m_f == it2.m_f);
}
/// operator !=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator!=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2)
{
    return (it1.m_base != it2.m_base) || (it1.m_f != it2.m_f);
}
/// operator <
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base < it2.m_base); }
/// operator <=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base <= it2.m_base); }
/// operator >
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base > it2.m_base); }
/// operator >=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base >= it2.m_base); }

///@} Iterators
///@} Basic

} // namespace nvbio
