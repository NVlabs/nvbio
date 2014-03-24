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

#include <nvbio/sufsort/sufsort_priv.h>
#include <nvbio/basic/string_set.h>
#include <nvbio/basic/thrust_view.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace nvbio {

///@addtogroup Sufsort
///@{

/// base virtual interface used by all string-set BWT handlers
///
struct BaseBWTHandler
{
    /// virtual destructor
    ///
    virtual ~BaseBWTHandler() {}

    /// process a batch of BWT symbols
    ///
    virtual void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices) {}
};

/// A class to output the BWT to a (potentially packed) device string
///
template <typename OutputIterator>
struct DeviceBWTHandler : public BaseBWTHandler
{
    /// constructor
    ///
    DeviceBWTHandler(OutputIterator _output) : output(_output), offset(0u) {}

    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        priv::device_copy(
            n_suffixes,
            d_bwt,
            output,
            offset );

        offset += n_suffixes;
    }

    OutputIterator               output;
    uint64                       offset;
};

/// A class to output the BWT to a host string
///
template <typename OutputIterator>
struct HostBWTHandler : public BaseBWTHandler
{
    HostBWTHandler(OutputIterator _output) : output(_output) {}

    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        for (uint32 i = 0; i < n_suffixes; ++i)
            output[i] = h_bwt[i];

        output += n_suffixes;
    }

    OutputIterator output;
};

/// A class to output the BWT to a packed host string
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename word_type>
struct HostBWTHandler< PackedStreamIterator< PackedStream<word_type*,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> > > : public BaseBWTHandler
{
    typedef PackedStreamIterator< PackedStream<word_type*,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> > OutputIterator;

    static const uint32 WORD_SIZE = uint32( 8u * sizeof(word_type) );
    static const uint32 SYMBOLS_PER_WORD = WORD_SIZE / SYMBOL_SIZE;


    /// constructor
    ///
    HostBWTHandler(OutputIterator _output) : output(_output), offset(0) {}

    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        const uint32 word_offset = offset & (SYMBOLS_PER_WORD-1);
              uint32 word_rem    = 0;
              uint32 word_idx    = offset / SYMBOLS_PER_WORD;

        word_type* words = output.container().stream();

        if (word_offset)
        {
            // compute how many symbols we still need to encode to fill the current word
            word_rem = SYMBOLS_PER_WORD - word_offset;

            // fetch the word in question
            word_type word = words[ word_idx ];

            for (uint32 i = 0; i < word_rem; ++i)
            {
                const uint32       bit_idx = (word_offset + i) * SYMBOL_SIZE;
                const uint32 symbol_offset = BIG_ENDIAN ? (WORD_SIZE - SYMBOL_SIZE - bit_idx) : bit_idx;
                const word_type     symbol = word_type(h_bwt[i]) << symbol_offset;

                // set bits
                word |= symbol;
            }

            // write out the word
            words[ word_idx ] = word;

            // advance word_idx
            ++word_idx;
        }

        for (uint32 i = word_rem; i < n_suffixes; i += SYMBOLS_PER_WORD)
        {
            // encode a word's worth of characters
            word_type word = 0u;

            const uint32 n_symbols = nvbio::min( SYMBOLS_PER_WORD, n_suffixes - i );

            for (uint32 j = 0; j < n_symbols; ++j)
            {
                const uint32       bit_idx = j * SYMBOL_SIZE;
                const uint32 symbol_offset = BIG_ENDIAN ? (WORD_SIZE - SYMBOL_SIZE - bit_idx) : bit_idx;
                const word_type     symbol = word_type(h_bwt[i + j]) << symbol_offset;

                // set bits
                word |= symbol;
            }

            // write out the word and advance word_idx
            words[ ++word_idx ] = word;
        }

        // advance the offset
        offset += n_suffixes;
    }

    OutputIterator output;
    uint64         offset;
};

/// A class to output the BWT to a packed host string
///
struct DiscardBWTHandler : public BaseBWTHandler
{
    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices) {}
};

///@}

} // namespace nvbio
