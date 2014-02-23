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

/*! \file sort.h
 *   \brief Define CUDA based sort primitives.
 */

#pragma once

#include <nvbio/basic/types.h>

namespace nvbio {
namespace cuda {

/// \page sorting_page Sorting
///
/// The SortEnactor provides a convenient wrapper around the fastest CUDA sorting library available,
/// allowing to perform both key-only and key-value pair sorting of arrays with the following
/// data-types:
///
/// - uint8
/// - uint16
/// - uint32
/// - uint64
/// - (uint8,uint32)
/// - (uint16,uint32)
/// - (uint32,uint32)
/// - (uint64,uint32)
///
///

///@addtogroup Basic
///@{

///@defgroup SortEnactors Sort Enactors
/// This module implements simple classes to sort device-memory buffers of key/value pairs of various primitive types.
///@{

/// A sorting buffer to hold vectors of key-value pairs
///
template <typename Keys, typename Values = null_type>
struct SortBuffers
{
    /// constructor
    ///
    SortBuffers() : selector(0) {}

    uint32  selector;
    Keys    keys[2];
    Values  values[2];
};

/// A simple class to enact sorts of various kinds
///
struct SortEnactor
{
    /// constructor
    ///
    SortEnactor();

    /// destructor
    ///
    ~SortEnactor();

    void sort(const uint32 count, SortBuffers<uint8*, uint32*>& buffers);
    void sort(const uint32 count, SortBuffers<uint16*,uint32*>& buffers);
    void sort(const uint32 count, SortBuffers<uint32*,uint32*>& buffers);
    void sort(const uint32 count, SortBuffers<uint32*,uint64*>& buffers);
    void sort(const uint32 count, SortBuffers<uint64*,uint32*>& buffers);
    void sort(const uint32 count, SortBuffers<uint8*>&          buffers);
    void sort(const uint32 count, SortBuffers<uint16*>&         buffers);
    void sort(const uint32 count, SortBuffers<uint32*>&         buffers);
    void sort(const uint32 count, SortBuffers<uint64*>&         buffers);

private:
    void*  m_impl;
};

///@} SortEnactors
///@} Basic

} // namespace cuda
} // namespace nvbio
