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

#include <nvbio/basic/packedstream.h>
#include <nvbio/alignment/sink.h>
#include <nvbio/alignment/utils.h>
#include <nvbio/alignment/alignment_base_inl.h>
#include <nvbio/basic/iterator.h>


namespace nvbio {
namespace aln {

namespace priv
{

///@addtogroup private
///@{

// -------------------------- Basic Hamming Distance functions ---------------------------- //

///
/// Calculate the alignment score between a pattern and a text, using the Smith-Waterman algorithm.
///
/// \tparam BAND_LEN            internal band length
/// \tparam TYPE                the alignment type
/// \tparam symbol_type         type of string symbols
///
template <AlignmentType TYPE, typename symbol_type>
struct ham_alignment_score_dispatch
{
    /// entry point
    ///
    /// \param context       template context class, used to specialize the behavior of the aligner
    /// \param query         input pattern (horizontal string)
    /// \param quals         input pattern qualities (horizontal string)
    /// \param ref           input text (vertical string)
    /// \param scoring       scoring scheme
    /// \param min_score     minimum output score
    /// \param sink          alignment sink
    /// \param window_begin  beginning of pattern window
    /// \param window_end    end of pattern window
    ///
    template <
        typename string_type,
        typename qual_type,
        typename ref_type,
        typename scoring_type,
        typename sink_type>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    static
    bool run(
        const scoring_type& scoring,
        string_type         query,
        qual_type           quals,
        ref_type            ref,
        const int32         min_score,
              sink_type&    sink)
    {
        const uint32 M = query.length();
        const uint32 N = ref.length();

        typedef int32 score_type;

        const score_type zero = score_type(0);

        for (uint32 i = 0; i + M < N; ++i)
        {
            score_type score = zero;

            for (uint32 j = 0; j < M; ++j)
            {
                const symbol_type r = ref[i+j];
                const symbol_type q = query[j];
                const symbol_type qq = quals[j];

                score += (q == r) ? scoring.match( qq ) : scoring.mismatch( r, q, qq );

                if (TYPE == LOCAL)
                {
                    score = nvbio::max( score, score_type(0) );
                    sink.report( score, make_uint2( i+j+1, j+1 ) );
                }
            }

            if (TYPE == SEMI_GLOBAL || TYPE == GLOBAL)
                sink.report( score, make_uint2( i+M, M ) );
        }
        return true;
    }
};

///
/// Calculate the alignment score between a pattern and a text, using the Smith-Waterman algorithm.
///
/// \tparam TYPE                the alignment type
/// \tparam pattern_string      pattern string 
/// \tparam quals_string        pattern qualities
/// \tparam text_string         text string
/// \tparam column_type         temporary column storage
///
template <
    AlignmentType   TYPE,
    typename        scoring_type,
    typename        algorithm_tag,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string,
    typename        column_type>
struct alignment_score_dispatch<
    HammingDistanceAligner<TYPE,scoring_type,algorithm_tag>,
    pattern_string,
    qual_string,
    text_string,
    column_type>
{
    typedef HammingDistanceAligner<TYPE,scoring_type,algorithm_tag> aligner_type;

    /// dispatch scoring across the whole pattern
    ///
    /// \param aligner      scoring scheme
    /// \param pattern      pattern string (horizontal
    /// \param quals        pattern qualities
    /// \param text         text string (vertical)
    /// \param min_score    minimum score
    /// \param sink         output alignment sink
    /// \param column       temporary column storage
    ///
    /// \return             true iff the minimum score was reached
    ///
    template <typename sink_type>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    static bool dispatch(
        const aligner_type      aligner,
        const pattern_string    pattern,
        const qual_string       quals,
        const text_string       text,
        const  int32            min_score,
              sink_type&        sink,
              column_type       column)
    {
        typedef typename pattern_string::value_type     symbol_type;

        return ham_alignment_score_dispatch<TYPE,symbol_type>::run( aligner.scheme, pattern, quals, text, min_score, sink );
    }
};

/// @} // end of private group

} // namespace priv

} // namespace aln
} // namespace nvbio
