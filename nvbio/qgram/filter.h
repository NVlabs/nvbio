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
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/cuda/primitives.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/exceptions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace nvbio {

///@addtogroup QGram
///@{

///
/// This class implements a q-gram filter which can be used to find and filter matches
/// between an arbitrary set of indexed query q-grams, representing q-grams of a given
/// text, and a \ref QGramIndex "q-gram index".
/// The q-gram index can be either a simple string index or a string-set index.
///
/// For string q-gram indices, the filter will return an ordered set of <i>(qgram-pos,query-pos)</i>
/// pairs, where <i>qgram-pos</i> is the index of the hit into the string used to build qgram-index,
/// and <i>query-pos</i> corresponds to one of the input query q-gram indices.
///
/// For string-set q-gram indices, the filter will return an ordered set of <i>(string-id,query-diagonal)</i>
/// pairs, where <i>string-id</i> is the index of the hit into the string-set used to build qgram-index, and
/// <i>query-diagonal</i> corresponds to the matching diagonal of the input query text.
///
struct QGramFilter
{
    /// enact the q-gram filter
    ///
    /// \param qgram_index      the q-gram index
    /// \param n_queries        the number of query q-grams
    /// \param queries          the query q-grams
    /// \param indices          the query indices
    ///
    template <typename qgram_index_type, typename query_iterator, typename index_iterator>
    void enact(
        const qgram_index_type& qgram_index,
        const uint32            n_queries,
        const query_iterator    queries,
        const index_iterator    indices);

    /// return the number of matching hits
    ///
    uint32 n_hits() const { return m_output.size(); }

    /// return the output list of hits
    ///
    const uint2* hits() const { return nvbio::plain_view( m_output ); }

    // TODO: generalize to the host
    thrust::device_vector<uint2>  m_ranges;
    thrust::device_vector<uint32> m_slots;
    thrust::device_vector<uint2>  m_output;
    thrust::device_vector<uint8>  d_temp_storage;
};

///@} // end of the QGram group

} // namespace nvbio

#include <nvbio/qgram/filter_inl.h>
