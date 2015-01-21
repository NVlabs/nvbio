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

///
///@addtogroup Alignment
///@{
///

///
///@addtogroup AlignmentSink Alignment Sinks
/// An alignment sink is an object passed to alignment functions to handle
/// the terminal (cell,score)- pairs of all valid alignments.
/// A particular sink might decide to discard or store all such alignments, while
/// another might decide to store only the best, or the best N, and so on.
///@{
///

///
/// A no-op sink for valid alignments
///
struct NullSink
{
    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    /// \param _sink     alignment's end
    ///
    template <typename ScoreType>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType _score, const uint2 _sink) {}
};

///
/// A sink for valid alignments, mantaining only a single best alignment
///
template <typename ScoreType>
struct BestSink
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    BestSink();

    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    /// \param _sink     alignment's end
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType _score, const uint2 _sink);

    ScoreType score;
    uint2     sink;
};

///
/// A sink for valid alignments, mantaining only a single best alignment
///
template <>
struct BestSink<simd4u8>
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    BestSink() : score( uint32(0) ) {}

    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const simd4u8 _score) { score = nvbio::max( score, _score ); }

    simd4u8 score;
};

///
/// A sink for valid alignments, mantaining the best two alignments
///
template <typename ScoreType>
struct Best2Sink
{
    /// constructor
    ///
    /// \param distinct_distance   the minimum text distance to consider two alignments distinct
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Best2Sink(const uint32 distinct_dist = 0);

    /// store a valid alignment
    ///
    /// \param score    alignment's score
    /// \param sink     alignment's end
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType score, const uint2 sink);

    ScoreType score1;
    ScoreType score2;
    uint2     sink1;
    uint2     sink2;

private:
    uint32    m_distinct_dist;
};

///
/// A sink for valid alignments, mantaining the best alignments by "column",
/// where columns have a specified width
///
template <typename ScoreType, uint32 N = 16>
struct BestColumnSink
{
    /// constructor
    ///
    /// \param column_width     the width of each column
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    BestColumnSink(const uint32 column_width = 50);

    /// store a valid alignment
    ///
    /// \param score    alignment's score
    /// \param sink     alignment's end
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType score, const uint2 sink);

    /// return the index of the best and second-best alignments
    ///
    void best2(uint32& i1, uint32& i2, const uint32 min_dist) const
    {
        ScoreType s1 = Field_traits<ScoreType>::min();
        ScoreType s2 = Field_traits<ScoreType>::min();
        i1 = N+1;
        i2 = N+1;

        // look for the best hit
        for (uint32 i = 0; i < N; ++i)
        {
            if (s1 < scores[i])
            {
                s1 = scores[i];
                i1 = i;
            }
        }

        // check whether we found a valid score
        if (s1 == Field_traits<ScoreType>::min())
            return;

        // look for the second hit, at a minimum distance from the best
        for (uint32 i = 0; i < N; ++i)
        {
            if ((s2 < scores[i]) &&
                ((sinks[i].x + min_dist <= sinks[i1].x) ||
                 (sinks[i].x            >= sinks[i1].x + min_dist)))
            {
                s2 = scores[i];
                i2 = i;
            }
        }
    }

    ScoreType scores[N+1];
    uint2     sinks[N+1];

private:
    uint32    m_column_width;
};

///@} // end of the AlignmentSink group

///@} // end Alignment group

} // namespace aln
} // namespace nvbio

#include <nvbio/alignment/sink_inl.h>
