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

namespace nvbio {

namespace qgram {

// return the size of a given range
struct range_size
{
    typedef uint2  argument_type;
    typedef uint32 result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 operator() (const uint2 range) const { return range.y - range.x; }
};

// given a (qgram-pos, text-pos) pair, return the closest regularly-spaced diagonal
struct closest_diagonal
{
    typedef uint2  argument_type;
    typedef uint2  result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    closest_diagonal(const uint32 _interval) : interval(_interval) {}

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const uint2 range) const
    {
        uint32 rounded_diag = util::round_z( range.y, interval );
        if (range.y - rounded_diag >= interval/2)
            rounded_diag++;

        return make_uint2( range.x, rounded_diag );
    }

    const uint32 interval;
};

template <typename qgram_index_type, typename index_iterator, typename coord_type>
struct filter_results {};

template <typename qgram_index_type, typename index_iterator>
struct filter_results< qgram_index_type, index_iterator, uint32 >
{
    typedef uint32  argument_type;
    typedef uint2   result_type;

    // constructor
    filter_results(
        const qgram_index_type  _qgram_index,
        const uint32            _n_queries,
        const uint32*           _slots,
        const uint2*            _ranges,
        const index_iterator    _index) :
    qgram_index ( _qgram_index ),
    n_queries   ( _n_queries ),
    slots       ( _slots ),
    ranges      ( _ranges ),
    index       ( _index ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 operator() (const uint32 output_index) const
    {
        // find the text q-gram slot corresponding to this output index
        const uint32 slot = uint32( upper_bound(
            output_index,
            slots,
            n_queries ) - slots );

        // fetch the corresponding text position
        const uint32 text_pos    = index[ slot ];

        // locate the hit q-gram position
        const uint2  range       = ranges[ slot ];
        const uint32 base_slot   = slot ? slots[ slot-1 ] : 0u;
        const uint32 local_index = output_index - base_slot;

        const uint32 qgram_pos = qgram_index.locate( range.x + local_index );

        // and write out the pair (qgram_pos,text_pos)
        return make_uint2( qgram_pos, text_pos );
    }

    const qgram_index_type  qgram_index;
    const uint32            n_queries;
    const uint32*           slots;
    const uint2*            ranges;
    const index_iterator    index;
};

template <typename qgram_index_type, typename index_iterator>
struct filter_results< qgram_index_type, index_iterator, uint2 >
{
    typedef uint32  argument_type;
    typedef uint2   result_type;

    // constructor
    filter_results(
        const qgram_index_type  _qgram_index,
        const uint32            _n_queries,
        const uint32*           _slots,
        const uint2*            _ranges,
        const index_iterator    _index) :
    qgram_index ( _qgram_index ),
    n_queries   ( _n_queries ),
    slots       ( _slots ),
    ranges      ( _ranges ),
    index       ( _index ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 operator() (const uint32 output_index) const
    {
        // find the text q-gram slot corresponding to this output index
        const uint32 slot = uint32( upper_bound(
            output_index,
            slots,
            n_queries ) - slots );

        // fetch the corresponding text position
        const uint32 text_pos    = index[ slot ];

        // locate the hit q-gram position
        const uint2  range       = ranges[ slot ];
        const uint32 base_slot   = slot ? slots[ slot-1 ] : 0u;
        const uint32 local_index = output_index - base_slot;

        const uint2 qgram_pos = qgram_index.locate( range.x + local_index );

        // and write out the pair (string-id,text-diagonal)
        return make_uint2( qgram_pos.x, text_pos - qgram_pos.y );
    }

    const qgram_index_type  qgram_index;
    const uint32            n_queries;
    const uint32*           slots;
    const uint2*            ranges;
    const index_iterator    index;
};

} // namespace qgram 

// enact the q-gram filter
//
// \param qgram_index      the q-gram index
// \param n_queries        the number of query q-grams
// \param queries          the query q-grams
// \param indices          the query indices
//
template <typename qgram_index_type, typename query_iterator, typename index_iterator>
void QGramFilter<host_tag>::enact(
    const qgram_index_type& qgram_index,
    const uint32            n_queries,
    const query_iterator    queries,
    const index_iterator    indices)
{
    typedef typename qgram_index_type::coord_type coord_type;

    m_ranges.resize( n_queries );
    m_slots.resize( n_queries );

    // search the q-grams in the index, obtaining a set of ranges
    thrust::transform(
        queries,
        queries + n_queries,
        m_ranges.begin(),
        qgram_index );

    // scan their size to determine the slots
    thrust::inclusive_scan(
        thrust::make_transform_iterator( m_ranges.begin(), qgram::range_size() ),
        thrust::make_transform_iterator( m_ranges.begin(), qgram::range_size() ) + n_queries,
        m_slots.begin() );

    // determine the total number of occurrences
    n_occurrences = m_slots[ n_queries-1 ];

    // resize the output buffer
    m_output.resize( n_occurrences );

    // and fill it
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_occurrences,
        m_output.begin(),
        qgram::filter_results<qgram_index_type,index_iterator,coord_type>(
            qgram_index,
            n_queries,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_ranges ),
            indices ) );
}

// merge hits falling within the same diagonal interval
//
void QGramFilter<host_tag>::merge(const uint32 interval)
{
    // snap the diagonals to the closest one
    thrust::transform(
        m_output.begin(),
        m_output.begin() + n_occurrences,
        m_output.begin(),
        qgram::closest_diagonal( interval ) );

    // now sort the results by (id, diagonal)
    uint64* output_ptr( (uint64*)nvbio::plain_view( m_output ) );
    thrust::sort(
        output_ptr,
        output_ptr + n_occurrences );

    // and run-length encode them
    thrust::host_vector<uint2> output( n_occurrences );

    m_counts.resize( n_occurrences );

    n_occurrences = uint32( thrust::reduce_by_key(
        m_output.begin(),
        m_output.begin() + n_occurrences,
        thrust::make_constant_iterator<uint32>(1u),
        output.begin(),
        m_counts.begin() ).first - output.begin() );

    // swap the outputs
    m_output.swap( output );

    // and shrink the output vectors
    m_output.resize( n_occurrences );
    m_counts.resize( n_occurrences );
}

// enact the q-gram filter
//
// \param qgram_index      the q-gram index
// \param n_queries        the number of query q-grams
// \param queries          the query q-grams
// \param indices          the query indices
//
template <typename qgram_index_type, typename query_iterator, typename index_iterator>
void QGramFilter<device_tag>::enact(
    const qgram_index_type& qgram_index,
    const uint32            n_queries,
    const query_iterator    queries,
    const index_iterator    indices)
{
    typedef typename qgram_index_type::coord_type coord_type;

    m_ranges.resize( n_queries );
    m_slots.resize( n_queries );

    // search the q-grams in the index, obtaining a set of ranges
    thrust::transform(
        queries,
        queries + n_queries,
        m_ranges.begin(),
        qgram_index );

    // scan their size to determine the slots
    cuda::inclusive_scan(
        n_queries,
        thrust::make_transform_iterator( m_ranges.begin(), qgram::range_size() ),
        m_slots.begin(),
        thrust::plus<uint32>(),
        d_temp_storage );

    // determine the total number of occurrences
    n_occurrences = m_slots[ n_queries-1 ];

    // resize the output buffer
    m_output.resize( n_occurrences );

    // and fill it
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_occurrences,
        m_output.begin(),
        qgram::filter_results<qgram_index_type,index_iterator,coord_type>(
            qgram_index,
            n_queries,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_ranges ),
            indices ) );
}

// merge hits falling within the same diagonal interval
//
void QGramFilter<device_tag>::merge(const uint32 interval)
{
    // snap the diagonals to the closest one
    thrust::transform(
        m_output.begin(),
        m_output.begin() + n_occurrences,
        m_output.begin(),
        qgram::closest_diagonal( interval ) );

    // now sort the results by (id, diagonal)
    thrust::device_ptr<uint64> output_ptr( (uint64*)nvbio::plain_view( m_output ) );
    thrust::sort(
        output_ptr,
        output_ptr + n_occurrences );

    // and run-length encode them
    thrust::device_vector<uint2> output( n_occurrences );

    m_counts.resize( n_occurrences );

    n_occurrences = cuda::runlength_encode(
        n_occurrences,
        m_output.begin(),
        output.begin(),
        m_counts.begin(),
        d_temp_storage );

    // swap the outputs
    m_output.swap( output );

    // and shrink the output vectors
    m_output.resize( n_occurrences );
    m_counts.resize( n_occurrences );
}

} // namespace nvbio
