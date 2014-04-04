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

#include <nvbio/qgram/qgram.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/cuda/primitives.h>
#include <nvbio/basic/thrust_view.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scatter.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

///\page qgroup_page Q-Group Index Module
///\htmlonly
/// <img src="nvidia_cubes.png" style="position:relative; bottom:-10px; border:0px;"/>
///\endhtmlonly
///\par
/// This module contains a series of functions to build a Q-Group Index, as described
/// in: \n
///
/// <i>Massively parallel read mapping on GPUs with PEANUT</i> \n
/// Johannes Koester and Sven Rahmann \n
/// http://arxiv.org/pdf/1403.1706v1.pdf
///
///
/// \section TechnicalOverviewSection Technical Overview
///\par
/// A complete list of the classes and functions in this module is given in the \ref QGroupIndex documentation.
///

namespace nvbio {

///
///@defgroup QGroupIndex Q-Group Index Module
/// This module contains a series of functions to build a Q-Group Index, as described
/// in: \n
///
/// <i>Massively parallel read mapping on GPUs with PEANUT</i> \n
/// Johannes Koester and Sven Rahmann \n
/// http://arxiv.org/pdf/1403.1706v1.pdf
///@{
///

/// A plain view of a q-group index (see \ref QGroupIndex)
///
struct QGroupIndexView
{
    static const uint32 WORD_SIZE = 32;

    typedef uint32*                                             vector_type;
    typedef PackedStream<const uint32*,uint8,1u,false,int64>    const_bitstream_type;
    typedef PackedStream<uint32*,uint8,1u,false,int64>          bitstream_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    QGroupIndexView(
        const uint32 _Q                 = 0,
        const uint32 _n_unique_qgrams   = 0,
        vector_type  _I                 = NULL,
        vector_type  _S                 = NULL,
        vector_type  _SS                = NULL,
        vector_type  _P                 = NULL) :
        Q               (_Q),
        n_unique_qgrams (_n_unique_qgrams),
        I               (_I),
        S               (_S),
        SS              (_SS),
        P               (_P)    {}

    /// return the slots of P corresponding to the given qgram g
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 range(const uint64 g) const
    {
        const uint32 i = uint32( g / WORD_SIZE );
        const uint32 j = uint32( g % WORD_SIZE );

        // check whether the j-th bit of I[i] is set
        if ((I[i] & (1u << j)) == 0u)
            return make_uint2( 0u, 0u );

        // compute j' such that bit j is the j'-th set bit in I[i]
        const uint32 j_prime = popc( I[i] & ((2u << j) - 1u) );

        return make_uint2(
            SS[ S[i] + j_prime ],
            SS[ S[i] + j_prime + 1u ] );
    }

    uint32        Q;
    uint32        n_unique_qgrams;
    vector_type   I;
    vector_type   S;
    vector_type   SS;
    vector_type   P;
};

/// A host-side q-group index (see \ref QGroupIndex)
///
struct QGroupIndexHost
{
    static const uint32 WORD_SIZE = 32;

    typedef thrust::host_vector<uint32>                         vector_type;
    typedef PackedStream<const uint32*,uint8,1u,false,int64>    const_bitstream_type;
    typedef PackedStream<uint32*,uint8,1u,false,int64>          bitstream_type;
    typedef QGroupIndexView                                     view_type;

    /// return the amount of device memory used
    ///
    uint64 used_host_memory() const
    {
        return I.size() * sizeof(uint32) +
               S.size() * sizeof(uint32) +
               SS.size() * sizeof(uint32) +
               P.size() * sizeof(uint32);
    }

    /// return the amount of device memory used
    ///
    uint64 used_device_memory() const { return 0u; }

    uint32        Q;
    uint32        n_unique_qgrams;
    vector_type   I;
    vector_type   S;
    vector_type   SS;
    vector_type   P;
};

/// A device-side q-group index (see \ref QGroupIndex)
///
struct QGroupIndexDevice
{
    static const uint32 WORD_SIZE = 32;

    typedef thrust::device_vector<uint32>                       vector_type;
    typedef PackedStream<const uint32*,uint8,1u,false,int64>    const_bitstream_type;
    typedef PackedStream<uint32*,uint8,1u,false,int64>          bitstream_type;
    typedef QGroupIndexView                                     view_type;

    /// build a q-group index from a given string T; the amount of storage required
    /// is basically O( A^q + |T|*32 ) bits, where A is the alphabet size.
    ///
    /// \tparam SYMBOL_SIZE     the size of the symbols, in bits
    /// \tparam string_type     the string iterator type
    ///
    /// \param q                the q parameter
    /// \param string_len       the size of the string
    /// \param string           the string iterator
    ///
    template <uint32 SYMBOL_SIZE, typename string_type>
    void build(
        const uint32        q,
        const uint32        string_len,
        const string_type   string);

    /// return the amount of device memory used
    ///
    uint64 used_host_memory() const { return 0u; }

    /// return the amount of device memory used
    ///
    uint64 used_device_memory() const
    {
        return I.size() * sizeof(uint32) +
               S.size() * sizeof(uint32) +
               SS.size() * sizeof(uint32) +
               P.size() * sizeof(uint32);
    }

    uint32        Q;
    uint32        n_unique_qgrams;
    vector_type   I;
    vector_type   S;
    vector_type   SS;
    vector_type   P;
};

/// return the plain view of a QGroupIndex
///
QGroupIndexView plain_view(QGroupIndexDevice& qgroup)
{
    return QGroupIndexView(
        qgroup.Q,
        qgroup.n_unique_qgrams,
        nvbio::plain_view( qgroup.I ),
        nvbio::plain_view( qgroup.S ),
        nvbio::plain_view( qgroup.SS ),
        nvbio::plain_view( qgroup.P ) );
}

///@} // end of the QGroupIndex group

} // namespace nvbio

#include <nvbio/qgram/qgroup_inl.h>
