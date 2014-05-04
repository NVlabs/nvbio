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

namespace nvbio {

//
// Build the occurrence table for a given string, packing a set of counters
// every K elements.
// The table must contain ((n+K-1)/K)*4 entries.
//
// Optionally save the table of the global counters as well.
//
// \param begin    symbol sequence begin
// \param end      symbol sequence end
// \param occ      output occurrence map
// \param cnt      optional table of the global counters
//
template <uint32 K, typename SymbolIterator, typename IndexType>
void build_occurrence_table(
    SymbolIterator begin,
    SymbolIterator end,
    IndexType*     occ,
    IndexType*     cnt)
{
    IndexType counters[4] = { 0u, 0u, 0u, 0u };

    const IndexType n = end - begin;

    for (IndexType i = 0; i < n; ++i)
    {
        const IndexType i_mod_K = i & (K-1);

        if (i_mod_K == 0)
        {
            // save the counters
            const uint32 k = i / K;
            for (uint32 c = 0; c < 4; ++c)
                occ[ k*4 + c ] = counters[c];
        }

        // update counters
        ++counters[ begin[i] ];
    }

    if (cnt)
    {
        // build a cumulative table of the final counters
        for (uint32 i = 0; i < 4; ++i)
            cnt[i] = counters[i];
    }
}

//
// TODO: CUDA build_occurrence_table
//
//	1. load 4 * 128 ints per cta in smem
//	have each thread reduce/scan 4 ints (=64bps)
//	have each warp do 2 2-short packed warp reduction/scans
//	have 1 warp do the final block-wide reduction
//	save block counters
//
//	2. scan blocks
//
//	3. redo 1. using block values
//

// sum a uint4 and a uchar4 packed into a uint32.
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void unpack_add(uint4* op1, const uint32 op2)
{
    op1->x += (op2       & 0xff);
    op1->y += (op2 >> 8  & 0xff);
    op1->z += (op2 >> 16 & 0xff);
    op1->w += (op2 >> 24);
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void unpack_add(uint64_4* op1, const uint32 op2)
{
    op1->x += (op2       & 0xff);
    op1->y += (op2 >> 8  & 0xff);
    op1->z += (op2 >> 16 & 0xff);
    op1->w += (op2 >> 24);
}

namespace occ {

// overload popc_2bit and popc_2bit_all so that they look the same
template <typename T> NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 mask, const T c)                      { return nvbio::popc_2bit_all( mask, c ); }
template <typename T> NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 mask, const T c,      const uint32 i) { return nvbio::popc_2bit_all( mask, c, i ); }
template <>           NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 mask, const uint32 c)                 { return nvbio::popc_2bit( mask, c ); }
template <>           NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 mask, const uint32 c, const uint32 i) { return nvbio::popc_2bit( mask, c, i ); }

// overload popc_2bit and popc_2bit_all so that they look the same
template <typename T> NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint64 mask, const T c)                      { return nvbio::popc_2bit_all( mask, c ); }
template <typename T> NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint64 mask, const T c,      const uint32 i) { return nvbio::popc_2bit_all( mask, c, i ); }
template <>           NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint64 mask, const uint32 c)                 { return nvbio::popc_2bit( mask, c ); }
template <>           NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint64 mask, const uint32 c, const uint32 i) { return nvbio::popc_2bit( mask, c, i ); }

// pop-count all the occurrences of c in each of the 32-bit masks in text[begin, end],
// where the last mask is truncated to i.
//
template <typename TextString, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(
    const TextString    text,
    const T             c,
    const uint32        begin,
    const uint32        end)
{
    uint32 x = 0;
    for (uint32 j = begin; j < end; ++j)
        x += occ::popc_2bit( text[j], c );

    return x;
}
// pop-count all the occurrences of c in each of the 32-bit masks in text[begin, end],
// where the last mask is truncated to i.
//
template <typename TextString, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(
    const TextString    text,
    const T             c,
    const uint32        begin,
    const uint32        end,
    const uint32        i)
{
    uint32 x = 0;
    for (uint32 j = begin; j < end; ++j)
        x += occ::popc_2bit( text[j], c );

    return x + occ::popc_2bit( text[ end ], c, i );
}

// pop-count all the occurrences of c in each of the 32-bit masks in text[begin, end],
// where the last mask is truncated to i.
//
template <typename TextString, typename T, typename W>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(
    const TextString    text,
    const T             c,
    const uint32        begin,
    const uint32        end,
    const uint32        i,
          W&            last_mask)
{
    uint32 x = 0;
    for (uint32 j = begin; j < end; ++j)
        x += occ::popc_2bit( text[j], c );

    last_mask = text[ end ];
    return x + occ::popc_2bit( last_mask, c, i );
}

} // namespace occ

// fetch the text character at position i in the rank dictionary
//
template <uint32 SYMBOL_SIZE_T, uint32 K, typename TextString, typename OccIterator, typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 text(const rank_dictionary<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable>& dict, const uint32 i)
{
    return dict.text[i];
}

// fetch the text character at position i in the rank dictionary
//
template <uint32 SYMBOL_SIZE_T, uint32 K, typename TextString, typename OccIterator, typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 text(const rank_dictionary<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable>& dict, const uint64 i)
{
    return dict.text[i];
}

template <uint32 SYMBOL_SIZE_T, uint32 K, typename TextString, typename OccIterator, typename CountTable, typename WordType, typename OccType>
struct dispatch_rank {};

template <typename T>
struct rank_word_traits {};

template <>
struct rank_word_traits<uint32>
{
    static const uint32 LOG_SYMS_PER_WORD = 4;
    static const uint32 SYMS_PER_WORD     = 16;
};
template <>
struct rank_word_traits<uint64>
{
    static const uint32 LOG_SYMS_PER_WORD = 5;
    static const uint32 SYMS_PER_WORD     = 32;
};

template <uint32 K, typename TextStorage, typename OccIterator, typename CountTable, typename word_type, typename index_type>
struct dispatch_rank<2,K,PackedStream<TextStorage,uint8,2u,true,index_type>,OccIterator,CountTable,word_type,index_type>
{
    typedef PackedStream<TextStorage,uint8,2u,true,index_type>      text_type;
    typedef rank_dictionary<2,K,text_type,OccIterator,CountTable>   dictionary_type;

    typedef typename vector_type<index_type,2>::type                vec2_type;
    typedef typename vector_type<index_type,4>::type                vec4_type;
    typedef vec2_type                                               range_type;

    static const uint32 LOG_SYMS_PER_WORD = rank_word_traits<word_type>::LOG_SYMS_PER_WORD;
    static const uint32 SYMS_PER_WORD     = rank_word_traits<word_type>::SYMS_PER_WORD;

    // pop-count the occurrences of symbols in two given text blocks, switching
    // between pop-counting a single symbol if T is a uint32, or all 4 symbols
    // if T is the count-table.
    template <typename T>
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint2 popc2(
        const TextStorage   text,
        const range_type    range,
        const uint32        kl,
        const uint32        kh,
        const T             c)
    {
        const uint32 ml = (range.x - kl*K) >> LOG_SYMS_PER_WORD;
        const uint32 mh = (range.y - kh*K) >> LOG_SYMS_PER_WORD;

        const word_type l_mod = ~word_type(range.x) & (SYMS_PER_WORD-1);
        const word_type h_mod = ~word_type(range.y) & (SYMS_PER_WORD-1);

        const uint32 offl = kl*(K >> LOG_SYMS_PER_WORD);

        // sum up all the pop-counts of the relevant masks, up to ml-1
        uint32 xl = occ::popc_2bit( text, c, offl, offl + ml );

        // check whether we can share the base value for the entire range
        uint32 xh     = (kl == kh) ? xl   : 0u;
        uint32 startm = (kl == kh) ? ml+1 : 0u;

        const word_type mask = text[ offl + ml ];
        xl += occ::popc_2bit( mask, c, l_mod );

        // if the range fits in a single block, add the last mask to xh
        if (kl == kh)
            xh += occ::popc_2bit( mask, c, (mh == ml) ? h_mod : 0u );

        // finish computing the end of the range
        if (kl != kh || mh > ml)
        {
            const uint32 offh = kh*(K >> LOG_SYMS_PER_WORD);
            xh += occ::popc_2bit( text, c, offh + startm, offh + mh, h_mod );
        }
        return make_uint2( xl, xh );
    }

    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type run(const dictionary_type& dict, const index_type i, const uint32 c)
    {
        if (i == index_type(-1))
            return 0u;

        const uint32 k = uint32( i / K );
        const uint32 m = (i - k*K) >> LOG_SYMS_PER_WORD;
        const word_type i_mod = ~word_type(i) & (SYMS_PER_WORD-1);

        // fetch base occurrence counter
        const word_type out = dict.occ[ k*4 + c ];

        const uint32 off = k*(K >> LOG_SYMS_PER_WORD);

        // sum up all the pop-counts of the relevant masks
        return out + occ::popc_2bit( dict.text.stream(), c, off, off + m, i_mod );
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE vec2_type run(const dictionary_type& dict, const range_type range, const uint32 c)
    {
        // check whether we're searching the empty range
        if (range.x == index_type(-1) && range.y == index_type(-1))
            return make_vector( index_type(0), index_type(0) );

        // check whether we're searching for one index only
        if (range.x == index_type(-1) || range.x == range.y)
        {
            const index_type r = run( dict, range.y, c );
            return make_vector( r, r );
        }

        const uint32 kl = uint32( range.x / K );
        const uint32 kh = uint32( range.y / K );

        // fetch base occurrence counters for the respective blocks
        const index_type outl = dict.occ[ kl*4 + c ];
        const index_type outh = (kl == kh) ? outl : dict.occ[ kh*4 + c ];

        const uint2 r = popc2( dict.text.stream(), range, kl, kh, c );

        return make_vector( outl + r.x, outh + r.y );
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE vec4_type run4(const dictionary_type& dict, const index_type i)
    {
        const uint32 k = uint32( i / K );
        const uint32 m = (i - k*K) >> LOG_SYMS_PER_WORD;

        // fetch base occurrence counters for all symbols in the respective block
        vec4_type r = make_vector( dict.occ[k*4+0], dict.occ[k*4+1], dict.occ[k*4+2], dict.occ[k*4+3] );

        const uint32 off = k*(K >> LOG_SYMS_PER_WORD);
        const uint32 x = occ::popc_2bit( dict.text.stream(), dict.count_table, off, off + m, ~word_type(i) & (SYMS_PER_WORD-1) );

        // add the packed counters to the output result
        unpack_add( &r, x );
        return r;
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void run4(const dictionary_type& dict, const range_type range, vec4_type* outl, vec4_type* outh)
    {
        const uint32 kl = uint32( range.x / K );
        const uint32 kh = uint32( range.y / K );

        // fetch base occurrence counters for for all symbols in the respective blocks
        *outl =                      make_vector( dict.occ[kl*4+0], dict.occ[kl*4+1], dict.occ[kl*4+2], dict.occ[kl*4+3] );
        *outh = (kl == kh) ? *outl : make_vector( dict.occ[kh*4+0], dict.occ[kh*4+1], dict.occ[kh*4+2], dict.occ[kh*4+3] );

        const uint2 r = popc2( dict.text.stream(), range, kl, kh, dict.count_table );

        // add the packed counters to the output result
        unpack_add( outl, r.x );
        unpack_add( outh, r.y );
    }
};

template <typename TextStorage, typename OccIterator, typename CountTable>
struct dispatch_rank<2,64,PackedStream<TextStorage,uint8,2u,true>,OccIterator,CountTable,uint4,uint4>
{
    static const uint32 LOG_K = 6;
    static const uint32 K     = 64;

    typedef PackedStream<TextStorage,uint8,2u,true>               text_type;
    typedef rank_dictionary<2,K,text_type,OccIterator,CountTable> dictionary_type;

    // pop-count the occurrences of symbols in a given text block, switching
    // between pop-counting a single symbol if T is a uint32, or all 4 symbols
    // if T is the count-table.
    template <typename T>
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc(
        const TextStorage   text,
        const uint32        i,
        const uint32        k,
        const T             c)
    {
        const uint32 m = (i - k*K) >> 4u;
        const uint32 i_16 = ~i&15;

        // sum up all the pop-counts of the relevant masks
        const uint4 masks = text[ k ];
        uint32 x = 0;
        if (m > 0) x += occ::popc_2bit( masks.x, c );
        if (m > 1) x += occ::popc_2bit( masks.y, c );
        if (m > 2) x += occ::popc_2bit( masks.z, c );

        // fetch the m-th mask and pop-count its nucleotides only up to i_mod_16
        return x + occ::popc_2bit( comp( masks, m ), c, i_16 );
    }

    // pop-count the occurrences of symbols in two given text blocks, switching
    // between pop-counting a single symbol if T is a uint32, or all 4 symbols
    // if T is the count-table.
    template <typename T>
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint2 popc2(
        const TextStorage   text,
        const uint2         range,
        const uint32        kl,
        const uint32        kh,
        const T             c)
    {
        const uint32 ml = (range.x - kl*K) >> 4u;
        const uint32 mh = (range.y - kh*K) >> 4u;

        const uint32 l_16 = ~range.x & 15;
        const uint32 h_16 = ~range.y & 15;

        // fetch the 128-bit text masks for the respective blocks
        const uint4 masks_l = text[ kl ];
        const uint4 masks_h = (kl == kh) ? masks_l : text[ kh ];

        // sum up all the pop-counts of the relevant masks
        uint32 xl = 0u;
        if (ml > 0) xl += occ::popc_2bit( masks_l.x, c );
        if (ml > 1) xl += occ::popc_2bit( masks_l.y, c );
        if (ml > 2) xl += occ::popc_2bit( masks_l.z, c );

        uint32 xh     = (kl == kh) ? xl : 0u;
        uint32 startm = (kl == kh) ? ml : 0u;

        // sum up all the pop-counts of the relevant masks
        if (mh > 0 && startm == 0) xh += occ::popc_2bit( masks_h.x, c );
        if (mh > 1 && startm <= 1) xh += occ::popc_2bit( masks_h.y, c );
        if (mh > 2 && startm <= 2) xh += occ::popc_2bit( masks_h.z, c );

        xl += occ::popc_2bit( comp( masks_l, ml ), c, l_16 );
        xh += occ::popc_2bit( comp( masks_h, mh ), c, h_16 );
        return make_uint2( xl, xh );
    }

    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 run(const dictionary_type& dict, const uint32 i, const uint32 c)
    {
        if (i == uint32(-1))
            return 0u;

        const uint32 k = i >> LOG_K;

        // fetch base occurrence counter
        const uint32 out = comp( dict.occ[ k ], c );

        return out + popc( dict.text.stream(), i, k, c );
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint2 run(const dictionary_type& dict, const uint2 range, const uint32 c)
    {
        // check whether we're searching the empty range
        if (range.x == uint32(-1) && range.y == uint32(-1))
            return make_uint2(0u,0u);

        // check whether we're searching for one index only
        if (range.x == uint32(-1) || range.x == range.y)
        {
            const uint32 r = run( dict, range.y, c );
            return make_uint2( range.x == uint32(-1) ? 0u : r, r );
        }

        const uint32 kl = range.x >> LOG_K;
        const uint32 kh = range.y >> LOG_K;

        // fetch the base occurrence counter at the beginning of the low block
        const uint32 outl = comp( dict.occ[ kl ], c );
        const uint32 outh = (kl == kh) ? outl : comp( dict.occ[ kh ], c );

        const uint2 r = popc2( dict.text.stream(), range, kl, kh, c );

        return make_uint2( outl + r.x, outh + r.y );
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint4 run4(const dictionary_type& dict, const uint32 i)
    {
        const uint32 k = i >> LOG_K;

        // fetch the base occurrence counter at the beginning of the block
        uint4 r = dict.occ[k];

        const uint32 x = popc( dict.text.stream(), i, k, dict.count_table );

        // add the packed counters to the output result
        unpack_add( &r, x );
        return r;
    }
    // fetch the number of occurrences of character c in the substring [0,i]
    static NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void run4(const dictionary_type& dict, const uint2 range, uint4* outl, uint4* outh)
    {
        const uint32 kl = range.x >> LOG_K;
        const uint32 kh = range.y >> LOG_K;

        // fetch the base occurrence counters at the beginning of the respective blocks
        *outl = dict.occ[kl];
        *outh = (kl == kh) ? *outl : dict.occ[kh];

        const uint2 r = popc2( dict.text.stream(), range, kl, kh, dict.count_table );

        // add the packed counters to the output result
        unpack_add( outl, r.x );
        unpack_add( outh, r.y );
    }
};

// fetch the number of occurrences of character c in the substring [0,i]
template <uint32 SYMBOL_SIZE_T, uint32 K, typename TextString, typename OccIterator, typename CountTable, typename IndexType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE IndexType rank(
    const rank_dictionary<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable>& dict, const IndexType i, const uint32 c)
{
    typedef typename TextString::storage_type                      word_type;
    typedef typename std::iterator_traits<OccIterator>::value_type occ_type;

    return dispatch_rank<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable,word_type,occ_type>::run(
        dict, i, c );
}

// fetch the number of occurrences of character c in the substring [0,i]
template <uint32 SYMBOL_SIZE_T, uint32 K, typename TextString, typename OccIterator, typename CountTable, typename IndexType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename vector_type<IndexType,2>::type rank(
    const rank_dictionary<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable>& dict, const typename vector_type<IndexType,2>::type range, const uint32 c)
{
    typedef typename TextString::storage_type                      word_type;
    typedef typename std::iterator_traits<OccIterator>::value_type occ_type;

    return dispatch_rank<SYMBOL_SIZE_T,K,TextString,OccIterator,CountTable,word_type,occ_type>::run(
        dict, range, c );
}

// fetch the number of occurrences of character c in the substring [0,i]
template <uint32 K, typename TextString, typename OccIterator, typename CountTable, typename IndexType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename vector_type<IndexType,4>::type rank4(
    const rank_dictionary<2,K,TextString,OccIterator,CountTable>& dict, const IndexType i)
{
    typedef typename TextString::storage_type                      word_type;
    typedef typename std::iterator_traits<OccIterator>::value_type occ_type;

    return dispatch_rank<2,K,TextString,OccIterator,CountTable,word_type,occ_type>::run4(
        dict, i );
}

// fetch the number of occurrences of character c in the substring [0,i]
template <uint32 K, typename TextString, typename OccIterator, typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void rank4(
    const rank_dictionary<2,K,TextString,OccIterator,CountTable>& dict, const uint2 range, uint4* outl, uint4* outh)
{
    typedef typename TextString::storage_type                      word_type;
    typedef typename std::iterator_traits<OccIterator>::value_type occ_type;

    dispatch_rank<2,K,TextString,OccIterator,CountTable,word_type,occ_type>::run4(
        dict, range, outl, outh );
}

// fetch the number of occurrences of character c in the substring [0,i]
template <uint32 K, typename TextString, typename OccIterator, typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void rank4(
    const rank_dictionary<2,K,TextString,OccIterator,CountTable>& dict, const uint64_2 range, uint64_4* outl, uint64_4* outh)
{
    typedef typename TextString::storage_type                      word_type;
    typedef typename std::iterator_traits<OccIterator>::value_type occ_type;

    dispatch_rank<2,K,TextString,OccIterator,CountTable,word_type,occ_type>::run4(
        dict, range, outl, outh );
}

} // namespace nvbio
