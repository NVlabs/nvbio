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

namespace nvbio {

///
/// The supported sequence alphabet types
///
enum SequenceAlphabet
{
    DNA     = 0u,
    DNA_N   = 1u,
    PROTEIN = 2u
};

/// A traits class for SequenceAlphabet
///
template <SequenceAlphabet ALPHABET> struct SequenceAlphabetTraits {};

/// A traits class for DNA SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;
};
/// A traits class for DNA_N SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;
};
/// A traits class for Protein SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<PROTEIN>
{
    static const uint32 SYMBOL_SIZE  = 8;
    static const uint32 SYMBOL_COUNT = 24;
};

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 bits_per_symbol(const SequenceAlphabet alphabet)
{
    return alphabet == DNA     ? 2 :
           alphabet == DNA_N   ? 4 :
           alphabet == PROTEIN ? 8 :
           8u;
}

} // namespace nvbio
