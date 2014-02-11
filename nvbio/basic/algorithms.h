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

/*! \file algorithms.h
 *   \brief Defines some general purpose algorithms.
 */

#pragma once

#include <nvbio/basic/types.h>

namespace nvbio {

/// \page algorithms_page Algorithms
///
/// NVBIO provides a few basic algorithms which can be called either from the host or the device:
///
/// - find_pivot()
/// - lower_bound()
/// - upper_bound()
/// - merge()
/// - merge_sort()
///

/// find the first element in a sequence for which a given predicate evaluates to true
///
/// \param begin        sequence start iterator
/// \param n            sequence size
/// \param predicate    unary predicate
template <typename Iterator, typename Predicate>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator find_pivot(
    Iterator        begin,
    const uint32    n,
    const Predicate predicate)
{
    // check whether this segment contains only 0s or only 1s
    if (predicate( begin[0] ) == predicate( begin[n-1] ))
        return predicate( begin[0] ) ? begin + n : begin;

    // perform a binary search over the given range
    uint32 count = n;

    while (count > 0)
    {
        const uint32 count2 = count / 2;

        Iterator mid = begin + count2;

        if (predicate( *mid ) == false)
            begin = ++mid, count -= count2 + 1;
        else
            count = count2;
    }
	return begin;
}

/// find the lower bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator lower_bound(
    const Value     x,
    Iterator        begin,
    const uint32    n)
{
    // check whether this segment is all left or right of x
    if (x < begin[0])
        return begin;

    if (begin[n-1] < x)
        return begin + n;

    // perform a binary search over the given range
    uint32 count = n;

    while (count > 0)
    {
        const uint32 count2 = count / 2;

        Iterator mid = begin + count2;

        if (*mid < x)
            begin = ++mid, count -= count2 + 1;
        else
            count = count2;
    }
	return begin;
}

/// find the upper bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator upper_bound(
    const Value     x,
    Iterator        begin,
    const uint32    n)
{
    uint32 count = n;
 
    while (count > 0)
    {
        const uint32 step = count / 2;

        Iterator it = begin + step;

        if (!(x < *it))
        {
            begin = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return begin;
}
} // namespace nvbio
