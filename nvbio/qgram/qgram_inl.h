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

// build a q-group index from a given string
//
// \param q                the q parameter
// \param string_len       the size of the string
// \param string           the string iterator
//
template <typename string_type>
void QGramIndexDevice::build(
    const uint32        q,
    const uint32        symbol_sz,
    const uint32        string_len,
    const string_type   string,
    const uint32        qlut)
{
    thrust::device_vector<uint8> d_temp_storage;

    symbol_size = symbol_sz;
    Q           = q;
    QL          = qlut;
    QLS         = (Q - QL) * symbol_size;

    qgrams.resize( string_len );
    index.resize( string_len );

    thrust::device_vector<uint32> d_all_qgrams( string_len );

    // build the list of q-grams
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + string_len,
        d_all_qgrams.begin(),
        string_qgram_functor<string_type>( Q, symbol_size, string_len, string ) );

    // build the list of q-gram indices
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + string_len,
        index.begin() );

    // sort them
    thrust::sort_by_key(
        d_all_qgrams.begin(),
        d_all_qgrams.begin() + string_len,
        index.begin() );

    // copy only the unique q-grams and count them
    thrust::device_vector<uint32> d_counts( string_len + 1u );

    n_unique_qgrams = cuda::runlength_encode(
        string_len,
        d_all_qgrams.begin(),
        qgrams.begin(),
        d_counts.begin(),
        d_temp_storage );

    // now we know how many unique q-grams there are
    slots.resize( n_unique_qgrams + 1u );

    // scan the counts to get the slots
    cuda::exclusive_scan(
        n_unique_qgrams + 1u,
        d_counts.begin(),
        slots.begin(),
        thrust::plus<uint32>(),
        uint32(0),
        d_temp_storage );

    // shrink the q-gram vector
    qgrams.resize( n_unique_qgrams );

    const uint32 n_slots = slots[ n_unique_qgrams ];
    if (n_slots != string_len)
        throw runtime_error( "mismatching number of q-grams: inserted %u q-grams, got: %u\n" );

    //
    // build a LUT
    //

    if (QL)
    {
        const uint32 ALPHABET_SIZE = 1u << symbol_size;

        uint64 lut_size = 1;
        for (uint32 i = 0; i < QL; ++i)
            lut_size *= ALPHABET_SIZE;

        // build a set of spaced q-grams
        thrust::device_vector<uint64> lut_qgrams( lut_size );
        thrust::transform(
            thrust::make_counting_iterator<uint32>(0),
            thrust::make_counting_iterator<uint32>(0) + lut_size,
            lut_qgrams.begin(),
            shift_left<uint64>( QLS ) );

        // and now search them
        lut.resize( lut_size+1 );

        thrust::lower_bound(
            qgrams.begin(),
            qgrams.begin() + n_unique_qgrams,
            lut_qgrams.begin(),
            lut_qgrams.begin() + lut_size,
            lut.begin() );

        // and write a sentinel value
        lut[ lut_size ] = n_unique_qgrams;
    }
    else
        lut.resize(0);
}


/// A functor fetching the length of the i-th string in a set
///
template <typename string_set_type>
struct length_functor
{
    typedef uint32 argument_type;
    typedef uint32 result_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    length_functor(const uint32 _Q, const string_set_type _string_set) : Q(_Q), string_set(_string_set) {}

    /// return the length of the i-th string, rounded to Q
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 operator() (const uint32 i) const { return util::round_z( string_set[i].length(), Q ); }

    const uint32          Q;
    const string_set_type string_set;
};


/// A functor to localize a string-set index
///
template <typename string_set_type>
struct localize_functor
{
    typedef uint32 argument_type;
    typedef uint2  result_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    localize_functor(const string_set_type _string_set, const uint32* _cum_lengths) :
        string_set(_string_set), cum_lengths(_cum_lengths) {}

    /// return the length of the i-th string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 operator() (const uint32 global_idx) const
    {
        const uint32 string_id = uint32( upper_bound( global_idx, cum_lengths, string_set.size() ) - cum_lengths );

        const uint32 base_offset = string_id ? cum_lengths[ string_id-1 ] : 0u;

        return make_uint2( string_id, global_idx - base_offset );
    }

    const string_set_type   string_set;
    const uint32*           cum_lengths;
};

// build a q-group index from a given string set
//
// \param q                the q parameter
// \param string           the string iterator
//
template <typename string_set_type>
void QGramSetIndexDevice::build(
    const uint32            q,
    const uint32            symbol_sz,
    const string_set_type   string_set,
    const uint32            qlut)
{
    thrust::device_vector<uint8> d_temp_storage;

    symbol_size = symbol_sz;
    Q           = q;
    QL          = qlut;
    QLS         = (Q - QL) * symbol_size;

    const uint32 n_strings = string_set.size();

    thrust::device_vector<uint32> cum_lengths( n_strings );

    // scan the string lengths, rounded to Q (to avoid partial q-grams with lots of zeroes)
    cuda::inclusive_scan(
        n_strings,
        thrust::make_transform_iterator( thrust::make_counting_iterator<uint32>(0u), length_functor<string_set_type>( Q, string_set ) ),
        cum_lengths.begin(),
        thrust::plus<uint32>(),
        d_temp_storage );

    const uint32 n_qgrams = cum_lengths[ n_strings-1 ];

    qgrams.resize( n_qgrams );
    index.resize( n_qgrams );

    thrust::device_vector<uint32> d_all_qgrams( n_qgrams );

    // build the list of q-gram indices
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_qgrams,
        index.begin(),
        localize_functor<string_set_type>( string_set, nvbio::plain_view( cum_lengths ) ) );

    // build the list of q-grams
    thrust::transform(
        index.begin(),
        index.begin() + n_qgrams,
        d_all_qgrams.begin(),
        string_set_qgram_functor<string_set_type>( Q, symbol_size, string_set ) );

    // sort them
    thrust::sort_by_key(
        d_all_qgrams.begin(),
        d_all_qgrams.begin() + n_qgrams,
        index.begin() );

    // copy only the unique q-grams and count them
    thrust::device_vector<uint32> d_counts( n_qgrams + 1u );

    n_unique_qgrams = cuda::runlength_encode(
        n_qgrams,
        d_all_qgrams.begin(),
        qgrams.begin(),
        d_counts.begin(),
        d_temp_storage );

    // now we know how many unique q-grams there are
    slots.resize( n_unique_qgrams + 1u );

    // scan the counts to get the slots
    cuda::exclusive_scan(
        n_unique_qgrams + 1u,
        d_counts.begin(),
        slots.begin(),
        thrust::plus<uint32>(),
        uint32(0),
        d_temp_storage );

    // shrink the q-gram vector
    qgrams.resize( n_unique_qgrams );

    const uint32 n_slots = slots[ n_unique_qgrams ];
    if (n_slots != n_qgrams)
        throw runtime_error( "mismatching number of q-grams: inserted %u q-grams, got: %u\n" );

    //
    // build a LUT
    //

    if (QL)
    {
        const uint32 ALPHABET_SIZE = 1u << symbol_size;

        uint64 lut_size = 1;
        for (uint32 i = 0; i < QL; ++i)
            lut_size *= ALPHABET_SIZE;

        // build a set of spaced q-grams
        thrust::device_vector<uint64> lut_qgrams( lut_size );
        thrust::transform(
            thrust::make_counting_iterator<uint32>(0),
            thrust::make_counting_iterator<uint32>(0) + lut_size,
            lut_qgrams.begin(),
            shift_left<uint64>( QLS ) );

        // and now search them
        lut.resize( lut_size+1 );

        thrust::lower_bound(
            qgrams.begin(),
            qgrams.begin() + n_unique_qgrams,
            lut_qgrams.begin(),
            lut_qgrams.begin() + lut_size,
            lut.begin() );

        // and write a sentinel value
        lut[ lut_size ] = n_unique_qgrams;
    }
    else
        lut.resize(0);
}

} // namespace nvbio
