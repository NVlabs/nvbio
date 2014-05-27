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
#include <nvbio/basic/dna.h>

namespace nvbio {

/// convert a Protein symbol to its ASCII character
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char protein_to_char(const uint8 c)
{
    return c ==  0 ? 'A' :
           c ==  1 ? 'C' :
           c ==  2 ? 'D' :
           c ==  3 ? 'E' :
           c ==  4 ? 'F' :
           c ==  5 ? 'G' :
           c ==  6 ? 'H' :
           c ==  7 ? 'I' :
           c ==  8 ? 'K' :
           c ==  9 ? 'L' :
           c == 10 ? 'M' :
           c == 11 ? 'N' :
           c == 12 ? 'P' :
           c == 13 ? 'Q' :
           c == 14 ? 'R' :
           c == 15 ? 'S' :
           c == 16 ? 'T' :
           c == 17 ? 'V' :
           c == 18 ? 'W' :
           c == 19 ? 'Y' :
           c == 20 ? 'B' :
           c == 21 ? 'Z' :
           c == 22 ? 'X' :
                     'N';
}

/// convert an ASCII character to a Protein symbol
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 char_to_protein(const char c)
{
    return c == 'A' ? 0u  :
           c == 'C' ? 1u  :
           c == 'D' ? 2u  :
           c == 'E' ? 3u  :
           c == 'F' ? 4u  :
           c == 'G' ? 5u  :
           c == 'H' ? 6u  :
           c == 'I' ? 7u  :
           c == 'K' ? 8u  :
           c == 'L' ? 9u  :
           c == 'M' ? 10u :
           c == 'N' ? 11u :
           c == 'P' ? 12u :
           c == 'Q' ? 13u :
           c == 'R' ? 14u :
           c == 'S' ? 15u :
           c == 'T' ? 16u :
           c == 'V' ? 17u :
           c == 'W' ? 18u :
           c == 'Y' ? 19u :
           c == 'B' ? 20u :
           c == 'Z' ? 21u :
           c == 'X' ? 22u :
                      11u;
}

// convert a given symbol to its ASCII character
//
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char to_char(const uint8 c)
{
    if (ALPHABET == DNA)
        return dna_to_char( c );
    else if (ALPHABET == DNA_N)
        return dna_to_char( c );
    else if (ALPHABET == IUPAC16)
        return iupac16_to_char( c );
    else if (ALPHABET == PROTEIN) // TODO!
        return protein_to_char( c );
}

// convert a given symbol to its ASCII character
//
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 from_char(const char c)
{
    if (ALPHABET == DNA)
        return char_to_dna( c );
    else if (ALPHABET == DNA_N)
        return char_to_dna( c );
    else if (ALPHABET == IUPAC16)
        return char_to_iupac16( c );
    else if (ALPHABET == PROTEIN) // TODO!
        return char_to_protein( c );
}

// convert from the given alphabet to an ASCII string
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const uint32         n,
    char*                string)
{
    for (uint32 i = 0; i < n; ++i)
        string[i] = to_char<ALPHABET>( begin[i] );

    string[n] = '\0';
}

// convert from the given alphabet to an ASCII string
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const SymbolIterator end,
    char*                string)
{
    for (SymbolIterator it = begin; it != end; ++it)
        string[ (it - begin) % (end - begin) ] = to_char<ALPHABET>( *it );

    string[ end - begin ] = '\0';
}

// convert from an ASCII string to the given alphabet
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const char*             end,
    const SymbolIterator    symbols)
{
    for (const char* it = begin; it != end; ++it)
        symbols[ (it - begin) % (end - begin) ] = from_char<ALPHABET>( *it );
}

// convert from an ASCII string to the given alphabet
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const SymbolIterator    symbols)
{
    for (const char* it = begin; *it != '\0'; ++it)
        symbols[ it - begin ] = from_char<ALPHABET>( *it );
}

} // namespace nvbio

