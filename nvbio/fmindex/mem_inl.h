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

// find all MEMs covering a given base
//
// \return the right-most position covered by a MEM
//
template <typename pattern_type, typename fm_index_type, typename delegate_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 find_kmems(
    const uint32            pattern_len,
    const pattern_type      pattern,
    const uint32            x,
    const fm_index_type     f_index,
    const fm_index_type     r_index,
          delegate_type&    handler,
    const uint32            min_intv,
    const uint32            min_span)
{
    typedef typename fm_index_type::index_type coord_type;
    typedef typename fm_index_type::range_type range_type;

    uint4 right_mems[1024];

    nvbio::vector_wrapper<uint4*> right_ranges( 0, &right_mems[0] );

    // find how far can we extend right starting from x
    //
    {
        // extend forward, using the reverse index
        range_type f_range = make_vector( coord_type(0u), f_index.length() );
        range_type r_range = make_vector( coord_type(0u), r_index.length() );

        range_type prev_range = f_range;

        uint32 i;
        for (i = x; i < pattern_len; ++i)
        {
            const uint8 c = pattern[i];
            if (c > 3) // there is an N here. no match
            {
                prev_range = f_range;
                break;
            }

            // search c in the FM-index
            extend_forward( f_index, r_index, f_range, r_range, c );

            // check if the range is too small
            if (1u + f_range.y - f_range.x < min_intv)
                break;

            // store the range
            if (f_range.y - f_range.x != prev_range.y - prev_range.x)
            {
                // do not add the empty span
                if (i > x)
                    right_ranges.push_back( make_vector( prev_range.x, prev_range.y, coord_type(x), coord_type(i) ) );

                prev_range = f_range;
            }
        }
        // check if we still need to save one range
        if (right_ranges.size() && (right_ranges.back().y - right_ranges.back().x) != (prev_range.y - prev_range.x))
            right_ranges.push_back( make_vector( prev_range.x, prev_range.y, coord_type(x), coord_type(i) ) );
    }

    // no valid match covering x
    if (right_ranges.size() == 0u)
        return x;

    // save the result value for later
    const uint32 rightmost_base = right_ranges.back().w;

    // now extend backwards, using the forward index
    //
    // we basically loop through all MEMs ending in [x,x+n_ranges) starting
    // from the end of the range and walking backwards - and for each of them we:
    //
    //  - find their starting point through backwards search
    //
    //  - and add them to the output only if they extend further left than
    //    any of the previously found ones
    //

    // keep track of the left-most coordinate covered by a MEM
    uint32 leftmost_coordinate = x+1;

    for (int32 j = right_ranges.size()-1; j >= 0; --j)
    {
              range_type  f_range = make_vector( right_ranges[j].x, right_ranges[j].y );
        const uint32      r       = right_ranges[j].w;

        // extend from r to the left as much possible
        const fm_index_type& index = f_index;

        int32 l;

        for (l = int32(x) - 1; l >= 0; --l)
        {
            const uint8 c = pattern[l];
            if (c > 3) // there is an N here. no match 
                break;

            // search c in the FM-index
            const range_type c_rank = rank(
                index,
                make_vector( f_range.x-1, f_range.y ),
                c );

            const range_type new_range = make_vector(
                index.L2(c) + c_rank.x + 1,
                index.L2(c) + c_rank.y );

            // stop if the range became too small
            if (1u + new_range.y - new_range.x < min_intv)
                break;

            // update the range
            f_range = new_range;
        }

        // only output the range if it's not contained in any other MEM
        if (uint32(l+1) < leftmost_coordinate && uint32(l+1) < r)
        {
            // save the range, together with its span
            const uint2 pattern_span = make_uint2( uint32(l+1), r );

            // keep the MEM only if it is above a certain length
            if (pattern_span.y - pattern_span.x >= min_span)
            {
                // pass all results to the delegate
                handler.output( f_range, pattern_span );

                // update the left-most covered coordinate
                leftmost_coordinate = uint32(l+1);
            }
        }
    }

    // return the right-most end of the MEMs covering x
    return rightmost_base;
}

// find all k-MEMs covering a given base, for each k
//
// \return the right-most position covered by a MEM
//
template <typename pattern_type, typename fm_index_type, typename delegate_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 find_threshold_kmems(
    const uint32            pattern_len,
    const pattern_type      pattern,
    const uint32            x,
    const fm_index_type     f_index,
    const fm_index_type     r_index,
          delegate_type&    handler,
    const uint32            min_intv,
    const uint32            min_span)
{
    typedef typename fm_index_type::index_type coord_type;
    typedef typename fm_index_type::range_type range_type;

    uint4 mems1[1024];
    uint4 mems2[1024];

    nvbio::vector_wrapper<uint4*> prev( 0, &mems1[0] );
    nvbio::vector_wrapper<uint4*> curr( 0, &mems2[0] );

    // find how far can we extend right starting from x
    //
    {
        // extend forward, using the reverse index
        range_type f_range = make_vector( coord_type(0u), f_index.length() );
        range_type r_range = make_vector( coord_type(0u), r_index.length() );

        range_type prev_range = f_range;

        uint32 i;
        for (i = x; i < pattern_len; ++i)
        {
            const uint8 c = pattern[i];
            if (c > 3) // there is an N here. no match
            {
                prev_range = f_range;
                break;
            }

            // search c in the FM-index
            extend_forward( f_index, r_index, f_range, r_range, c );

            // check if the range is too small
            if (1u + f_range.y - f_range.x < min_intv)
                break;

            // store the range
            if (f_range.y - f_range.x != prev_range.y - prev_range.x)
            {
                // do not add the empty span
                if (i > x)
                    curr.push_back( make_vector( prev_range.x, prev_range.y, coord_type(x), coord_type(i) ) );

                prev_range = f_range;
            }
        }
        // check if we still need to save one range
        if (curr.size() && (curr.back().y - curr.back().x) != (prev_range.y - prev_range.x))
            curr.push_back( make_vector( prev_range.x, prev_range.y, coord_type(x), coord_type(i) ) );

        // swap prev and curr
        nvbio::vector_wrapper<uint4*> tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // no valid match covering x
    if (prev.size() == 0u)
        return x;

    // save the result value for later
    const uint32 rightmost_base = prev.back().w;

    uint32 out_n = 0;
    uint32 out_i = x+1;

	for (int32 i = int32(x) - 1; i >= -1; --i)
    {
        // backward search for MEMs
		const uint8 c = i < 0 ? 4u : pattern[i]; // c > 3 if i < 0 or pattern[i] is an ambiguous base

        // reset the output buffer size
        curr.resize( 0 );

        for (int32 j = prev.size()-1; j >= 0; --j)
        {
            const range_type  f_range = make_vector( prev[j].x, prev[j].y );
            const uint32      r_span  = prev[j].w;

            // search c in the FM-index
            const range_type c_rank = c < 4u ?
                rank( f_index, make_vector( f_range.x-1, f_range.y ), c ) :
                make_vector( coord_type(0), coord_type(0) );

            const range_type new_range = make_vector(
                f_index.L2(c) + c_rank.x + 1,
                f_index.L2(c) + c_rank.y );

            // keep the hit if reaching the beginning or an ambiguous base or the intv is small enough
            if (c > 3u || c_rank.y - c_rank.x < min_intv)
            {
                // test curr_n > 0 to make sure there are no longer matches
				if (curr.size() == 0)
                {
                    if (out_n == 0 || i+1 < out_i) // skip contained matches
                    {
                        // save the range, together with its span
                        const uint2 pattern_span = make_uint2( i+1, r_span );

                        // keep the MEM only if it is above a certain length
                        if (pattern_span.y - pattern_span.x >= min_span)
                        {
                            // pass all results to the delegate
                            handler.output( f_range, pattern_span );

                            out_n++;
                            out_i = i+1;
                        }
                    }
				} // otherwise the match is contained in another longer match
            }
            else if (curr.size() == 0 || (new_range.y - new_range.x) != (curr.back().y - curr.back().x))
            {
                // save the range, together with its span
                curr.push_back( make_vector(
                    new_range.x,
                    new_range.y,
                    coord_type( i+1 ),
                    coord_type( r_span ) ) );
            }
		}
        // check whether there's no more work left
		if (curr.size() == 0)
            break;

        // swap prev and curr
        nvbio::vector_wrapper<uint4*> tmp = prev;
        prev = curr;
        curr = tmp;
    }

    // return the right-most end of the MEMs covering x
    return rightmost_base;
}

namespace mem {

// return the size of a given range
template <typename rank_type>
struct range_size
{
    typedef rank_type argument_type;
    typedef uint64    result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint64 operator() (const rank_type range) const { return 1u + range.y - range.x; }
};

// return the string-id of a given MEM rank
template <typename rank_type>
struct rank_string_id
{
    typedef rank_type argument_type;
    typedef uint32    result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 operator() (const rank_type range) const { return range.z; }
};

// return the span of a given range
template <typename rank_type>
struct rank_span_size
{
    typedef rank_type argument_type;
    typedef uint64    result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint64 operator() (const rank_type range) const
    {
        return (range.w >> 16u) - (range.w & 0xFFu);
    }
};

// a simple mem handler
template <typename coord_type>
struct mem_handler
{
    static const uint32 MAX_SIZE = 1024;

    typedef typename vector_type<coord_type,2u>::type range_type;
    typedef typename vector_type<coord_type,4u>::type mem_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_handler(const uint32 _string_id, const uint32 _max_intv) :
        string_id(_string_id),
        max_intv(_max_intv),
        n_mems(0) {}

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void output(const range_type range, const uint2 span)
    {
        if (n_mems >= MAX_SIZE)
            return;

        // check whether the SA range is small enough
        if (1u + range.y - range.x <= max_intv)
        {
            mems[ n_mems++ ] = make_vector(
                coord_type( range.x ),
                coord_type( range.y ),
                coord_type( string_id ),
                coord_type( span.x | span.y << 16u ) );
        }
    }

    const uint32    string_id;
    const uint32    max_intv;
    mem_type        mems[MAX_SIZE];
    uint32          n_mems;
};

template <MEMSearchType TYPE, typename index_type, typename string_set_type>
struct mem_functor
{
    typedef typename index_type::index_type             coord_type;
    typedef typename index_type::range_type             range_type;
    typedef typename vector_type<coord_type,4u>::type   mem_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_functor(
        const index_type        _f_index,
        const index_type        _r_index,
        const string_set_type   _string_set,
        const uint32            _min_intv,
        const uint32            _max_intv,
        const uint32            _min_span,
        VectorArrayView<uint4>  _mem_arrays) :
    f_index      ( _f_index ),
    r_index      ( _r_index ),
    string_set   ( _string_set ),
    min_intv     ( _min_intv ),
    max_intv     ( _max_intv ),
    min_span     ( _min_span ),
    mem_arrays   ( _mem_arrays ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void operator() (const uint32 string_id) const
    {
        // fetch the pattern
        typename string_set_type::string_type pattern = string_set[ string_id ];

        // compute its length
        const uint32 pattern_len = nvbio::length( pattern );

        // build a MEM handler
        mem_handler<coord_type> handler( string_id, max_intv );

        // and collect all MEMs
        for (uint32 x = 0; x < pattern_len;)
        {
            // find MEMs covering x and move to the next uncovered position along the pattern
            if (TYPE == THRESHOLD_KMEM_SEARCH)
            {
                const uint32 y = find_threshold_kmems(
                    pattern_len,
                    pattern,
                    x,
                    f_index,
                    r_index,
                    handler,
                    min_intv,
                    min_span );

                x = nvbio::max( y, x+1u );
            }
            else
            {
                const uint32 y = find_kmems(
                    pattern_len,
                    pattern,
                    x,
                    f_index,
                    r_index,
                    handler,
                    min_intv,
                    min_span );

                x = nvbio::max( y, x+1u );
            }
        }

        // output the array of results
        if (handler.n_mems)
        {
            mem_type* output = mem_arrays.alloc( string_id, handler.n_mems );
            if (output != NULL)
            {
                // output in reverse order, i.e. sorted by the starting coordinate
                for (uint32 i = 0; i < handler.n_mems; ++i)
                    output[i] = handler.mems[ handler.n_mems - i - 1u ];
            }
        }
    }

    const index_type                    f_index;
    const index_type                    r_index;
    const string_set_type               string_set;
    const uint32                        min_intv;
    const uint32                        max_intv;
    const uint32                        min_span;
    mutable VectorArrayView<mem_type>   mem_arrays;
};

template <typename index_type, typename string_set_type>
struct right_mem_functor
{
    typedef typename index_type::index_type             coord_type;
    typedef typename index_type::range_type             range_type;
    typedef typename vector_type<coord_type,4u>::type   mem_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    right_mem_functor(
        const index_type        _f_index,
        const index_type        _r_index,
        const string_set_type   _string_set,
        const uint32            _min_intv,
        const uint32            _max_intv,
        const uint32            _min_span,
        VectorArrayView<uint4>  _mem_arrays) :
    f_index      ( _f_index ),
    r_index      ( _r_index ),
    string_set   ( _string_set ),
    min_intv     ( _min_intv ),
    max_intv     ( _max_intv ),
    min_span     ( _min_span ),
    mem_arrays   ( _mem_arrays ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void operator() (const uint32 string_id) const
    {
        // fetch the pattern
        typename string_set_type::string_type pattern = string_set[ string_id ];

        // compute its length
        const uint32 pattern_len = nvbio::length( pattern );

        // build a MEM handler
        mem_type                    mems[1024];
        vector_wrapper<mem_type*>   mems_vec( 0u, mems );

        // and collect all MEMs
        for (uint32 x = 0; x < pattern_len;)
        {
            const uint32 y = right_kmems(
                pattern_len,
                pattern,
                string_id,
                x,
                mems_vec,
                min_intv );

            x = nvbio::max( y, x+1u );
        }

        // output the array of results
        if (mems_vec.size())
        {
            mem_type* output = mem_arrays.alloc( string_id, mems_vec.size() );
            if (output != NULL)
            {
                for (uint32 i = 0; i < mems_vec.size(); ++i)
                    output[i] = mems_vec[i];
            }
        }
    }

    // find all MEMs covering a given base
    //
    // \return the right-most position covered by a MEM
    //
    template <typename pattern_type, typename output_vector>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 right_kmems(
        const uint32            pattern_len,
        const pattern_type      pattern,
        const uint32            string_id,
        const uint32            x,
              output_vector&    output,
        const uint32            min_intv) const
    {
        // find how far can we extend right starting from x
        //

        // internal output
        mem_type                    mems[1024];
        vector_wrapper<mem_type*>   mems_vec( 0u, mems );

        // extend forward, using the reverse index
        range_type f_range = make_vector( coord_type(0u), f_index.length() );
        range_type r_range = make_vector( coord_type(0u), r_index.length() );

        range_type prev_range = f_range;

        uint32 i;
        for (i = x; i < pattern_len; ++i)
        {
            const uint8 c = pattern[i];
            if (c > 3) // there is an N here. no match
            {
                prev_range = f_range;
                break;
            }

            // search c in the FM-index
            extend_forward( f_index, r_index, f_range, r_range, c );

            // check if the range is too small
            if (1u + f_range.y - f_range.x < min_intv)
                break;

            // store the range
            if (f_range.y - f_range.x != prev_range.y - prev_range.x)
            {
                // do not add the empty span
                if (i > x)
                    mems_vec.push_back( make_vector( prev_range.x, prev_range.y, string_id, coord_type(x) | (coord_type(i) << 16) ) );

                prev_range = f_range;
            }
        }
        // check if we still need to save one range
        if (mems_vec.size() && (mems_vec.back().y - mems_vec.back().x) != (prev_range.y - prev_range.x))
            mems_vec.push_back( make_vector( prev_range.x, prev_range.y, string_id, coord_type(x) | (coord_type(i) << 16) ) );

        // no valid match covering x
        if (mems_vec.size() == 0u)
            return x;

        const uint32 MEM_GROUP_MARKER = 1u << 31;

        // output them in reverse order, i.e. from largest to smallest
        for (uint32 i = 0; i < mems_vec.size(); ++i)
        {
            mem_type mem = mems_vec[ mems_vec.size() - i - 1u ];

            // add a special marker to identify the rightmost MEM covering a given position later on
            if (i == 0)
                mem.z |= MEM_GROUP_MARKER;

            output.push_back( mem );
        }

        // return the rightmost coordinate
        return (mems_vec.back().w >> 16);
    }

    const index_type                    f_index;
    const index_type                    r_index;
    const string_set_type               string_set;
    const uint32                        min_intv;
    const uint32                        max_intv;
    const uint32                        min_span;
    mutable VectorArrayView<mem_type>   mem_arrays;
};

// A functor to extend a MEM to the left
//
template <typename index_type, typename string_set_type>
struct left_mem_functor
{
    typedef typename index_type::index_type             coord_type;
    typedef typename index_type::range_type             range_type;
    typedef typename vector_type<coord_type,4u>::type   mem_type;

    // functor typedefs
    typedef mem_type    argument_type;
    typedef mem_type    result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    left_mem_functor(
        const index_type        _f_index,
        const index_type        _r_index,
        const string_set_type   _string_set,
        const uint32            _min_intv,
        const uint32            _max_intv,
        const uint32            _min_span) :
    f_index      ( _f_index ),
    r_index      ( _r_index ),
    string_set   ( _string_set ),
    min_intv     ( _min_intv ),
    max_intv     ( _max_intv ),
    min_span     ( _min_span ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_type operator() (const mem_type mem) const
    {
        const uint32 MEM_GROUP_MARKER = 1u << 31;

              range_type    f_range   = make_vector( mem.x, mem.y ); 
        const uint32        string_id = mem.z & (~MEM_GROUP_MARKER); // remove the special marker
        const uint32        x         = mem.w & 0xFFu;
        const uint32        r         = mem.w >> 16;

        // fetch the pattern
        typename string_set_type::string_type pattern = string_set[ string_id ];

        // extend from x to the left as much possible
        const index_type& index = f_index;

        int32 l;

        for (l = int32(x) - 1; l >= 0; --l)
        {
            const uint8 c = pattern[l];
            if (c > 3) // there is an N here. no match 
                break;

            // search c in the FM-index
            const range_type c_rank = rank(
                index,
                make_vector( f_range.x-1, f_range.y ),
                c );

            const range_type new_range = make_vector(
                index.L2(c) + c_rank.x + 1,
                index.L2(c) + c_rank.y );

            // stop if the range became too small
            if (1u + new_range.y - new_range.x < min_intv)
                break;

            // update the range
            f_range = new_range;
        }

        return make_vector(
            f_range.x,
            f_range.y,
            mem.z,
            coord_type(l+1) | coord_type(r << 16) );
    }

    const index_type                    f_index;
    const index_type                    r_index;
    const string_set_type               string_set;
    const uint32                        min_intv;
    const uint32                        max_intv;
    const uint32                        min_span;
};

template <typename coord_type>
struct filter_results
{
    typedef typename vector_type<coord_type,2u>::type   range_type;
    typedef typename vector_type<coord_type,4u>::type   rank_type;
    typedef MEMHit<coord_type>                          mem_type;

    typedef rank_type  argument_type;
    typedef mem_type   result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    filter_results(
        const uint32        _n_ranges,
        const uint64*       _slots,
        const rank_type*    _ranges) :
    n_ranges    ( _n_ranges ),
    slots       ( _slots ),
    ranges      ( _ranges ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_type operator() (const uint64 output_index) const
    {
        // find the text q-gram slot corresponding to this output index
        const uint32 slot = uint32( upper_bound(
            output_index,
            slots,
            n_ranges ) - slots );

        // locate the hit position
        const rank_type range    = ranges[ slot ];
        const uint64 base_slot   = slot ? slots[ slot-1 ] : 0u;
        const uint32 local_index = output_index - base_slot;

        // and write out the MEM occurrence
        return mem_type(
            coord_type( range.x + local_index ),    // SA coordinate for this occurrence
            coord_type( 0u ),                       // unused
            range.z,                                // string-id
            range.w );                              // packed pattern span
    }

    const uint32        n_ranges;
    const uint64*       slots;
    const rank_type*    ranges;
};


template <typename index_type>
struct locate_ssa_results
{
    typedef typename index_type::index_type             coord_type;
    typedef typename index_type::range_type             range_type;
    typedef MEMHit<coord_type>                          mem_type;

    typedef mem_type    argument_type;
    typedef mem_type    result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    locate_ssa_results(const index_type _index) : index( _index ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_type operator() (const mem_type mem) const
    {
        const range_type r = locate_ssa_iterator( index, mem.coords.x );

        return mem_type(
            coord_type( r.x ),
            coord_type( r.y ),
            mem.coords.z,
            mem.coords.w );
    }

    const index_type index;
};

template <typename index_type>
struct lookup_ssa_results
{
    typedef typename index_type::index_type             coord_type;
    typedef typename index_type::range_type             range_type;
    typedef MEMHit<coord_type>                          mem_type;

    typedef mem_type    argument_type;
    typedef mem_type    result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    lookup_ssa_results(const index_type _index) : index( _index ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    mem_type operator() (const mem_type mem) const
    {
        const coord_type loc = lookup_ssa_iterator( index, make_vector( mem.coords.x, mem.coords.y ) );
        return mem_type(
            loc,
            mem.coords.z,
            coord_type( mem.coords.w & 0xFFu ),
            coord_type( mem.coords.w >> 16u ) );
    }

    const index_type index;
};


// device kernel to reorder a vector array of mem-ranges
template <typename rank_type>
__global__
void discard_ranges_kernel(
    const uint32                            n_items,        // # of input items
          VectorArrayView<rank_type>        ranges,         // input vector array
    const uint32                            max_intv,       // max SA interval size
    const uint32                            min_span)       // minimum pattern span
{
    const uint32 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_items)
        return;

    // fetch the i-th vector and discard any MEMs which don't pass our search criteria
    const uint32     vec_size = ranges.size( i );
          rank_type* vec      = ranges[i];

    // keep track of the leftmost MEM coordinate
    uint32 leftmost_coordinate = uint32(-1);

    uint32 group_begin = 0u;
    uint32 n_output    = 0u;

    for (uint32 j = 0; j < vec_size; ++j)
    {
        const uint32 MEM_GROUP_MARKER = 1u << 31;

        rank_type rank = vec[j];

        // check the special marker to see if this is the initial MEM for a group
        if (rank.z & MEM_GROUP_MARKER)
        {
            // reverse the order of the MEMs in the previous group, so so as to sort them by left coordinate
            for (uint32 k = 0; k < (n_output - group_begin)/2; ++k)
            {
                const rank_type tmp      = vec[ group_begin + k ];
                vec[ group_begin + k ]   = vec[ n_output - k - 1u ];
                vec[ n_output - k - 1u ] = tmp;
            }

            // remove the marker
            rank.z &= ~MEM_GROUP_MARKER;

            // reset the leftmost MEM coordinate
            leftmost_coordinate = uint32(-1);

            // reset the group start
            group_begin = n_output;
        }

        const uint2 range = make_uint2( rank.x, rank.y );
        const uint2 span  = make_uint2( rank.w & 0xFFu, rank.w >> 16 );

        // check whether we want to keep this MEM rank
        if (span.x < leftmost_coordinate &&             // make sure it's not contained in a previous MEM
            span.y - span.x >= min_span  &&             // make sure its span is long enough
            1u + range.y - range.x <= max_intv)         // make sure it doesn't have too many occurrences
        {
            vec[ n_output++ ] = rank;

            // update the leftmost coordinate
            leftmost_coordinate = span.x;
        }
    }

    // write out the new vector size
    ranges.m_sizes[i] = n_output;
}

// device function to reorder a vector array of mem-ranges
template <typename rank_type>
void discard_ranges(
    const device_tag                        system_tag,     // system tag
    const uint32                            n_items,        // # of input items
          VectorArrayView<rank_type>        ranges,         // input vector array
    const uint32                            max_intv,       // max SA interval size
    const uint32                            min_span)       // minimum pattern span
{
    const uint32 block_dim = 128;
    const uint32 n_blocks = util::divide_ri( n_items, block_dim );

    discard_ranges_kernel<<<n_blocks,block_dim>>>(
        n_items,
        ranges,
        max_intv,
        min_span );
}

// copy the array of ranges bound to index i into the proper position of the output
template <typename rank_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void copy_ranges(
    const uint32                            i,              // vector index to copy
    const VectorArrayView<rank_type>        in_ranges,      // input vector array
    const uint32*                           slots,          // output slots
    rank_type*                              out_ranges)     // output arena
{
    const uint32 slot = slots[i];

    const uint32     n_src = in_ranges.size( i );
    const rank_type* src   = in_ranges[i];
          rank_type* dst   = out_ranges + slot;

    for (uint32 j = 0; j < n_src; ++j)
        dst[j] = src[j];
}

// device kernel to reorder a vector array of mem-ranges
template <typename rank_type>
__global__
void reorder_ranges_kernel(
    const uint32                            n_items,        // # of input items
    const VectorArrayView<rank_type>        in_ranges,      // input vector array
    const uint32*                           slots,          // output slots
    rank_type*                              out_ranges)     // output arena
{
    const uint32 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_items)
        return;

    copy_ranges( i, in_ranges, slots, out_ranges );
}

// device function to reorder a vector array of mem-ranges
template <typename rank_type>
void reorder_ranges(
    const device_tag                        system_tag,     // system tag
    const uint32                            n_items,        // # of input items
    const VectorArrayView<rank_type>        in_ranges,      // input vector array
    const uint32*                           slots,          // output slots
    rank_type*                              out_ranges)     // output arena
{
    const uint32 block_dim = 128;
    const uint32 n_blocks = util::divide_ri( n_items, block_dim );

    reorder_ranges_kernel<<<n_blocks,block_dim>>>(
        n_items,
        in_ranges,
        slots,
        out_ranges );
}
// host function to reorder a vector array of mem-ranges
template <typename rank_type>
void reorder_ranges(
    const host_tag                          system_tag,     // system tag
    const uint32                            n_items,        // # of input items
    const VectorArrayView<rank_type>        in_ranges,      // input vector array
    const uint32*                           slots,          // output slots
    rank_type*                              out_ranges)     // output arena
{
    #pragma omp parallel for
    for (int i = 0; i < int( n_items ); ++i)
        copy_ranges( i, in_ranges, slots, out_ranges );
}

} // namespace mem


// enact the filter on an FM-index and a string-set
//
// \param fm_index         the FM-index
// \param string-set       the query string-set
//
// \return the total number of hits
//
template <typename fm_index_type>
template <typename string_set_type>
uint64 MEMFilter<host_tag, fm_index_type>::rank(
    const MEMSearchType     search_type,
    const fm_index_type&    f_index,
    const fm_index_type&    r_index,
    const string_set_type&  string_set,
    const uint32            min_intv,
    const uint32            max_intv,
    const uint32            min_span)
{
    // save the query
    m_n_queries     = string_set.size();
    m_f_index       = f_index;
    m_r_index       = r_index;
    m_n_occurrences = 0;

    const uint32 max_string_length = 256; // TODO: compute this

    m_mem_ranges.resize( m_n_queries, max_string_length * m_n_queries );

    // search the strings in the index, obtaining a set of ranges
    if (search_type == THRESHOLD_KMEM_SEARCH)
    {
        thrust::for_each(
            thrust::make_counting_iterator<uint32>(0u),
            thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
            mem::mem_functor<THRESHOLD_KMEM_SEARCH,fm_index_type,string_set_type>(
                m_f_index,
                m_r_index,
                string_set,
                min_intv,
                max_intv,
                min_span,
                nvbio::plain_view( m_mem_ranges ) )
            );
    }
    else
    {
        thrust::for_each(
            thrust::make_counting_iterator<uint32>(0u),
            thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
            mem::mem_functor<KMEM_SEARCH,fm_index_type,string_set_type>(
                m_f_index,
                m_r_index,
                string_set,
                min_intv,
                max_intv,
                min_span,
                nvbio::plain_view( m_mem_ranges ) )
            );
    }

    // fetch the number of output MEM ranges
    const uint32 n_ranges = m_mem_ranges.allocated_size();

    // reserve enough storage for the ranges
    m_slots.resize( n_ranges );

    if (n_ranges)
    {
        // reorder the arena by string-id
        thrust::host_vector<rank_type> arena( n_ranges );
        thrust::host_vector<uint32>    slots( m_n_queries );

        // scan the mem-range array sizes to get the new array slots
        thrust::exclusive_scan(
            m_mem_ranges.m_sizes.begin(),
            m_mem_ranges.m_sizes.begin() + m_n_queries,
            slots.begin() );

        // and put everything in place
        mem::reorder_ranges(
            host_tag(),
            m_n_queries,
            nvbio::plain_view( m_mem_ranges ),
            nvbio::plain_view( slots ),
            nvbio::plain_view( arena ) );

        // swap the new array indices and the arena
        m_mem_ranges.m_index.swap( slots );
        m_mem_ranges.m_arena.swap( arena );

        // and now scan the range sizes
        thrust::inclusive_scan(
            thrust::make_transform_iterator( m_mem_ranges.m_arena.begin(), mem::range_size<rank_type>() ),
            thrust::make_transform_iterator( m_mem_ranges.m_arena.begin(), mem::range_size<rank_type>() ) + n_ranges,
            m_slots.begin() );

        // fetch the total number of MEMs
        m_n_occurrences = m_slots[ n_ranges - 1u ];
    }
    return m_n_occurrences;
}

// find the starting position for the MEM ranks corresponding to a given string
//
template <typename fm_index_type>
uint32 MEMFilter<host_tag, fm_index_type>::first_hit(const uint32 string_id) const
{
    // fetch the total number of MEM ranges
    if (string_id >= m_n_queries)
        return m_n_occurrences;

    const uint32 first_rank = m_mem_ranges.m_index[ string_id ];

    // and find the corresponding MEM hits start
    return first_rank ? m_slots[ first_rank-1u ] : 0u;
}

// enumerate all mems in a given range
//
// \tparam mems_iterator         a mem_type iterator
//
// \param begin                  the beginning of the mems sequence to locate, in [0,n_mems)
// \param end                    the end of the mems sequence to locate, in [0,n_mems]
//
template <typename fm_index_type>
template <typename mems_iterator>
void MEMFilter<host_tag, fm_index_type>::locate(
    const uint64    begin,
    const uint64    end,
    mems_iterator   mems)
{
    const uint32 n_hits = end - begin;

    // fetch the number of output MEM ranges
    const uint32 n_ranges = m_mem_ranges.allocated_size();

    // fill the output hits with (SA,string-id) coordinates
    thrust::transform(
        thrust::make_counting_iterator<uint64>(0u) + begin,
        thrust::make_counting_iterator<uint64>(0u) + end,
        mems,
        mem::filter_results<coord_type>(
            n_ranges,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_mem_ranges.m_arena ) ) );

    // locate the SSA iterators
    thrust::transform(
        mems,
        mems + n_hits,
        mems,
        mem::locate_ssa_results<fm_index_type>( m_f_index ) );

    // and perform the final lookup
    thrust::transform(
        mems,
        mems + n_hits,
        mems,
        mem::lookup_ssa_results<fm_index_type>( m_f_index ) );
}

// enact the filter on an FM-index and a string-set
//
// \param fm_index         the FM-index
// \param string-set       the query string-set
//
// \return the total number of hits
//
template <typename fm_index_type>
template <typename string_set_type>
uint64 MEMFilter<device_tag, fm_index_type>::rank(
    const MEMSearchType     search_type,
    const fm_index_type&    f_index,
    const fm_index_type&    r_index,
    const string_set_type&  string_set,
    const uint32            min_intv,
    const uint32            max_intv,
    const uint32            min_span)
{
    // save the query
    m_n_queries     = string_set.size();
    m_f_index       = f_index;
    m_r_index       = r_index;
    m_n_occurrences = 0;

    const uint32 max_string_length = 256; // TODO: compute this

    m_mem_ranges.resize( m_n_queries, max_string_length * m_n_queries );

    // search the strings in the index, obtaining a set of ranges
    if (search_type == THRESHOLD_KMEM_SEARCH)
    {
        thrust::for_each(
            thrust::make_counting_iterator<uint32>(0u),
            thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
            mem::mem_functor<THRESHOLD_KMEM_SEARCH,fm_index_type,string_set_type>(
                m_f_index,
                m_r_index,
                string_set,
                min_intv,
                max_intv,
                min_span,
                nvbio::plain_view( m_mem_ranges ) )
            );
    }
    else
    {
        thrust::for_each(
            thrust::make_counting_iterator<uint32>(0u),
            thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
            mem::right_mem_functor<fm_index_type,string_set_type>(
                m_f_index,
                m_r_index,
                string_set,
                min_intv,
                max_intv,
                min_span,
                nvbio::plain_view( m_mem_ranges ) )
            );

        // fetch the number of output MEM ranges
        const uint32 n_ranges = m_mem_ranges.allocated_size();

        // extend them left
        thrust::transform(
            m_mem_ranges.m_arena.begin(),
            m_mem_ranges.m_arena.begin() + n_ranges,
            m_mem_ranges.m_arena.begin(),
            mem::left_mem_functor<fm_index_type,string_set_type>(
                m_f_index,
                m_r_index,
                string_set,
                min_intv,
                max_intv,
                min_span )
            );

        // and discard MEM ranges we don't want to keep
        mem::discard_ranges(
            device_tag(),
            m_n_queries,
            nvbio::plain_view( m_mem_ranges ),
            max_intv,
            min_span );
    }

    // fetch the number of output MEM ranges
    //const uint32 n_ranges = m_mem_ranges.allocated_size();

    // compute the number of output MEM ranges
    // NOTE: this reduction is necessary as discard_ranges might have changed the
    // number of ranges effectively in use
    const uint32 n_ranges = cuda::reduce(
        m_n_queries,
        m_mem_ranges.m_sizes.begin(),
        thrust::plus<uint32>(),
        d_temp_storage );

    // reset the pool size
    m_mem_ranges.m_pool[0] = n_ranges;

    // reserve enough storage for the ranges
    m_slots.resize( n_ranges );

    if (n_ranges)
    {
        // reorder the arena by string-id
        thrust::device_vector<rank_type> arena( n_ranges );
        thrust::device_vector<uint32>    slots( m_n_queries );

        // scan the mem-range array sizes to get the new array slots
        cuda::exclusive_scan(
            m_n_queries,
            m_mem_ranges.m_sizes.begin(),
            slots.begin(),
            thrust::plus<uint32>(),
            0u,
            d_temp_storage );

        // and put everything in place
        mem::reorder_ranges(
            device_tag(),
            m_n_queries,
            nvbio::plain_view( m_mem_ranges ),
            nvbio::plain_view( slots ),
            nvbio::plain_view( arena ) );

        // swap the new array indices and the arena
        m_mem_ranges.m_index.swap( slots );
        m_mem_ranges.m_arena.swap( arena );

        // and now scan the range sizes
        cuda::inclusive_scan(
            n_ranges,
            thrust::make_transform_iterator( nvbio::plain_view( m_mem_ranges.m_arena ), mem::range_size<rank_type>() ),
            m_slots.begin(),
            thrust::plus<uint64>(),
            d_temp_storage );

        // fetch the total number of MEMs
        m_n_occurrences = m_slots[ n_ranges - 1u ];
    }
    return m_n_occurrences;
}

// find the starting position for the MEM ranks corresponding to a given string
//
template <typename fm_index_type>
uint32 MEMFilter<device_tag, fm_index_type>::first_hit(const uint32 string_id) const
{
    // fetch the total number of MEM ranges
    if (string_id >= m_n_queries)
        return m_n_occurrences;

    const uint32 first_rank = m_mem_ranges.m_index[ string_id ];

    // and find the corresponding MEM hits start
    return first_rank ? m_slots[ first_rank-1u ] : 0u;

}

// enumerate all mems in a given range
//
// \tparam mems_iterator         a mem_type iterator
//
// \param begin                  the beginning of the mems sequence to locate, in [0,n_mems)
// \param end                    the end of the mems sequence to locate, in [0,n_mems]
//
template <typename fm_index_type>
template <typename mems_iterator>
void MEMFilter<device_tag, fm_index_type>::locate(
    const uint64    begin,
    const uint64    end,
    mems_iterator   mems)
{
    const uint32 n_hits = end - begin;

    // fetch the number of output MEM ranges
    const uint32 n_ranges = m_mem_ranges.allocated_size();

    // fill the output hits with (SA,string-id) coordinates
    thrust::transform(
        thrust::make_counting_iterator<uint64>(0u) + begin,
        thrust::make_counting_iterator<uint64>(0u) + end,
        device_iterator( mems ),
        mem::filter_results<coord_type>(
            n_ranges,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_mem_ranges.m_arena ) ) );

    // locate the SSA iterators
    thrust::transform(
        device_iterator( mems ),
        device_iterator( mems ) + n_hits,
        device_iterator( mems ),
        mem::locate_ssa_results<fm_index_type>( m_f_index ) );

    // and perform the final lookup
    thrust::transform(
        device_iterator( mems ),
        device_iterator( mems ) + n_hits,
        device_iterator( mems ),
        mem::lookup_ssa_results<fm_index_type>( m_f_index ) );
}

} // namespace nvbio
