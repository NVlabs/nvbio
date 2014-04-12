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

// qmap.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/alignment/alignment.h>
#include <nvbio/alignment/batched.h>
#include <nvbio/strings/string_set.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/utils.h>

using namespace nvbio;

enum { CACHE_SIZE = 64 };
typedef nvbio::lmem_cache_tag<CACHE_SIZE>                                       lmem_cache_tag_type;
typedef nvbio::uncached_tag                                                     uncached_tag_type;

//
// An alignment stream class to be used in conjunction with the BatchAlignmentScore class
//
template <uint32 BAND_LEN, typename t_aligner_type, typename cache_type = lmem_cache_tag_type>
struct AlignmentStream
{
    typedef t_aligner_type                                                              aligner_type;
    typedef io::ReadDataDevice::const_plain_view_type                                   read_view_type;

    typedef nvbio::cuda::ldg_pointer<uint32>                                            base_iterator;

    typedef ReadLoader<read_view_type,cache_type>                                       pattern_loader_type;
    typedef typename pattern_loader_type::string_type                                   pattern_string;

    typedef GenomeLoader<base_iterator,cache_type>                                      text_loader_type;
    typedef typename text_loader_type::string_type                                      text_string;

    // an alignment context
    struct context_type
    {
        uint2                   read_range;
        uint32                  genome_begin;
        uint32                  genome_end;
        int32                   min_score;
        aln::BestSink<int32>    sink;
    };
    // a container for the strings to be aligned
    struct strings_type
    {
        pattern_loader_type         pattern_loader;
        text_loader_type            text_loader;
        pattern_string              pattern;
        aln::trivial_quality_string quals;
        text_string                 text;
    };

    // constructor
    AlignmentStream(
        aligner_type            _aligner,
        const uint32            _count,
        const uint2*            _diagonals,
        const read_view_type    _reads,
        const uint32            _genome_len,
        const uint32*           _genome,
               int16*           _scores) :
        m_aligner           ( _aligner ),
        m_count             (_count),
        m_diagonals         (_diagonals),
        m_reads             (_reads),
        m_genome_len        (_genome_len),
        m_genome            (_genome),
        m_scores            (_scores) {}

    // get the aligner
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const aligner_type& aligner() const { return m_aligner; };

    // return the maximum pattern length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 max_pattern_length() const { return m_reads.max_read_len(); }

    // return the maximum text length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 max_text_length() const { return m_reads.max_read_len() + BAND_LEN; }

    // return the stream size
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_count; }

    // return the i-th pattern's length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pattern_length(const uint32 i, context_type* context) const
    {
        return context->read_range.y - context->read_range.x;
    }

    // return the i-th text's length
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 text_length(const uint32 i, context_type* context) const
    {
        return context->genome_end - context->genome_begin;
    }

    // initialize the i-th context
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool init_context(
        const uint32    i,
        context_type*   context) const
    {
        // fetch the diagonal for the i-th alignment task
        const uint2 diagonal  = m_diagonals[i];

        // retrieve the read id and the text position
        const uint32 read_id    = diagonal.y;
        const uint32 text_pos   = diagonal.x;

        // fetch the read range
        context->read_range   = m_reads.get_range( read_id );
        const uint32 read_len = context->read_range.y - context->read_range.x;

        // compute the segment of text to align to
        context->genome_begin = text_pos > BAND_LEN/2 ? text_pos - BAND_LEN/2 : 0u;
        context->genome_end   = nvbio::min( context->genome_begin + read_len + BAND_LEN, m_genome_len );

        // initialize the sink
        context->sink = aln::BestSink<int32>();

        // setup the minimum score
        context->min_score = Field_traits<int32>::min();
        return true;
    }

    // initialize the i-th context
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void load_strings(
        const uint32        i,
        const uint32        window_begin,
        const uint32        window_end,
        const context_type* context,
              strings_type* strings) const
    {
        const uint2 read_subrange = make_uint2( window_begin, window_end );

        // load the pattern
        strings->pattern = strings->pattern_loader.load( m_reads, context->read_range, FORWARD, STANDARD, read_subrange );

        // load the text
        strings->text = strings->text_loader.load( m_genome, context->genome_begin, context->genome_end - context->genome_begin );
    }

    // handle the output
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void output(
        const uint32        i,
        const context_type* context) const
    {
        // copy the output score
        m_scores[i] = context->sink.score;
    }

    aligner_type    m_aligner;
    uint32          m_count;
    const uint2*    m_diagonals;
    read_view_type  m_reads;
    const uint32    m_genome_len;
    base_iterator   m_genome;
    int16*          m_scores;
};
