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

/*! \file vector.h
 *   \brief Define host / device vectors
 */

#pragma once

#include <nvbio/basic/types.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace nvbio {

/// a dynamic host/device vector class
///
template <typename system_tag, typename T>
struct vector {};

/// a dynamic host vector class
///
template <typename T>
struct vector<host_tag,T> : public thrust::host_vector<T>
{
    typedef host_tag                            system_tag;

    typedef thrust::host_vector<T>              base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    /// constructor
    ///
    vector<host_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<host_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<host_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<host_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<host_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }
};

/// a dynamic device vector class
///
template <typename T>
struct vector<device_tag,T> : public thrust::device_vector<T>
{
    typedef device_tag                          system_tag;

    typedef thrust::device_vector<T>            base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    /// constructor
    ///
    vector<device_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<device_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<device_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<device_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<device_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }
};

} // namespace nvbio
