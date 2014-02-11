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
#include <nvbio/basic/simd.h>

namespace nvbio {
namespace aln {

// A sink for valid alignments, mantaining only a single best alignment
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
BestSink<ScoreType>::BestSink() : score( Field_traits<ScoreType>::min() ), sink( make_uint2( uint32(-1), uint32(-1) ) ) {}

// store a valid alignment
//
// \param score    alignment's score
// \param sink     alignment's end
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void BestSink<ScoreType>::report(const ScoreType _score, const uint2 _sink)
{
    // NOTE: we must use <= here because otherwise we won't pick the bottom-right most one
    // in case there's multiple optimal scores
    if (score <= _score)
    {
        score = _score;
        sink  = _sink;
    }
}

// A sink for valid alignments, mantaining the best two alignments
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Best2Sink<ScoreType>::Best2Sink(const uint32 distinct_dist) :
    score1( Field_traits<ScoreType>::min() ),
    score2( Field_traits<ScoreType>::min() ),
    sink1( make_uint2( uint32(-1), uint32(-1) ) ),
    sink2( make_uint2( uint32(-1), uint32(-1) ) ),
    m_distinct_dist( distinct_dist ) {}

// store a valid alignment
//
// \param score    alignment's score
// \param sink     alignment's end
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void Best2Sink<ScoreType>::report(const ScoreType score, const uint2 sink)
{
    // NOTE: we must use <= here because otherwise we won't pick the bottom-right most one
    // in case there's multiple optimal scores
    if (score1 <= score)
    {
        score1 = score;
        sink1  = sink;
    }
    else if (score2 <= score && (sink.x + m_distinct_dist < sink1.x || sink.x > sink1.x + m_distinct_dist))
    {
        score2 = score;
        sink2  = sink;
    }
}

} // namespace aln
} // namespace nvbio
