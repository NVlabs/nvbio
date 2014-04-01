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
#include <nvbio/basic/packedstream.h>
#include <nvbio/fmindex/rank_dictionary.h>

namespace nvbio {

///
/// \page fmindex_page FM-Index Module
///\htmlonly
/// <img src="nvidia_cubes.png" style="position:relative; bottom:-10px; border:0px;"/>
///\endhtmlonly
///
///\n
/// This module defines a series of classes to represent and operate on 2-bit FM-indices,
/// both from the host and device CUDA code.
///
/// \section RankDictionarySection Rank Dictionaries
///
/// A rank_dictionary is a data structure that, given a text T, allows to count the number
/// of occurrences of any given character c in a given prefix T[0,i] in O(1) time.
///
/// rank_dictionary is <i>storage-free</i>, in the sense it doesn't directly hold any allocated
/// data - hence it can be instantiated both on host and device data-structures, and it can
/// be conveniently passed as a kernel parameter (provided its template parameters are also
/// PODs).
///
/// It supports the following functions:
///
/// <table>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// <b>Function</b>
/// </td><td style="vertical-align:text-top;">
/// <b>Inputs</b>
/// </td><td style="vertical-align:text-top;">
/// <b>Description</b>
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank()<br>
/// </td><td style="vertical-align:text-top;">
/// dict, i, c
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of a given character c in the prefix [0,i]
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank()<br>
/// </td><td style="vertical-align:text-top;">
/// dict, range, c
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of a given character c in the prefixes [0,range.x] and [0,range.y]
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank4()<br>
/// </td><td style="vertical-align:text-top;">
/// dict, i
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of all characters of a 4-letter alphabet in the prefix [0,i]
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank4()<br>
/// </td><td style="vertical-align:text-top;">
/// dict, range
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of all characters of a 4-letter alphabet in the prefixes [0,range.x] and [0,range.y]
/// </td></tr>
/// </table>
///
/// \section SSASection Sampled Suffix Arrays
///
/// A <i>Sampled Suffix Array</i> is a succint suffix array which has been sampled at a subset of
/// its indices. Ferragina & Manzini showed that given such a data structure and the BWT of the
/// original text it is possible to reconstruct the missing locations.
/// Two such data structures have been proposed, with different tradeoffs:
///
///  - one storing only the entries that are a multiple of K, { SA[i] | SA[i] % K = 0 }
///  - one storing only the entries whose index is a multiple of K, { SA[i] | i % K = 0 }
///
/// NVBIO provides both:
///
///  - SSA_value_multiple
///  - SSA_index_multiple
///
/// Unlike for the rank_dictionary, which is a storage-free class, these classes own the (internal) storage
/// needed to represent the underlying data structures, which resides on the host.
/// Similarly, the module provides some counterparts that hold the corresponding storage for the device:
///
///  - SSA_value_multiple_device
///  - SSA_index_multiple_device
///
/// While these classes hold device data, they are meant to be used from the host and cannot be directly
/// passed to CUDA kernels.
/// Plain views (see \ref host_device_page), or <i>contexts</i>, can be obtained with the usual plain_view() function.
///
/// The contexts expose the following interface:
///
/// \code
/// struct SSA
/// {
///     // return the i-th suffix array entry, if present
///     bool fetch(const uint32 i, uint32& r) const
///
///     // return whether the i-th suffix array is present
///     bool has(const uint32 i) const
/// }
/// \endcode
///
/// Detailed documentation can be found in the \ref SSAModule module documentation.
///
/// \section FMIndexSection FM-Indices
///
/// An fm_index is a self-compressed text index as described by Ferragina & Manzini.
/// It is built on top of the following ingredients:
///     - the BWT of a text T
///     - a rank dictionary (\ref rank_dictionary) of the given BWT
///     - a sampled suffix array of T
///
/// Given the above, it allows to count and locate all the occurrences in T of arbitrary patterns
/// P in O(length(P)) time.
/// Moreover, it does so with an incremental algorithm that proceeds character by character,
/// a fundamental property that allows to implement sophisticated pattern matching algorithms
/// based on backtracking.
///
/// fm_index is <i>storage-free</i>, in the sense it doesn't directly hold any allocated
/// data - hence it can be instantiated both on host and device data-structures, and it can
/// be conveniently passed as a kernel parameter (provided its template parameters are also
/// PODs).
///
/// It supports the following functions:
///
/// <table>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// <b>Function</b>
/// </td><td style="vertical-align:text-top;">
/// <b>Inputs</b>
/// </td><td style="vertical-align:text-top;">
/// <b>Description</b>
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank()<br>
/// </td><td style="vertical-align:text-top;">
/// fmi, i, c
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of a given character c in the prefix [0,i]
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// rank4()<br>
/// </td><td style="vertical-align:text-top;">
/// fmi, i
/// </td><td style="vertical-align:text-top;">
/// return the number of occurrences of all characters of a 4-letter alphabet in the prefix [0,i]
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// match()<br>
/// </td><td style="vertical-align:text-top;">
/// fmi, pattern
/// </td><td style="vertical-align:text-top;">
/// return the SA range of occurrences of a given pattern
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// match_reverse()<br>
/// </td><td style="vertical-align:text-top;">
/// fmi, pattern
/// </td><td style="vertical-align:text-top;">
/// return the SA range of occurrences of a given reversed pattern
/// </td></tr>
/// </td></tr>
/// <tr><td style="white-space: nowrap; vertical-align:text-top;">
/// locate()<br>
/// </td><td style="vertical-align:text-top;">
/// fmi, i
/// </td><td style="vertical-align:text-top;">
/// given a suffix array coordinate i, return its linear coordinate SA[i]
/// </td></tr>
/// </table>
///
/// \section PerformanceSection Performance
///
/// The graphs below show the performance of exact and approximate matching using the FM-index on the CPU and GPU,
/// searching for 32bp fragments inside the whole human genome. Approximate matching is performed with the hamming_backtrack()
/// function, in this case allowing up to 1 mismatch per fragment:
///
/// <img src="benchmark-fm-index.png" style="position:relative; bottom:-10px; border:0px;" width="50%" height="50%"/>
/// <img src="benchmark-fm-index-speedup.png" style="position:relative; bottom:-10px; border:0px;" width="50%" height="50%"/>
///
/// \section TechnicalOverviewSection Technical Overview
///
/// A complete list of the classes and functions in this module is given in the \ref FMIndex documentation.
///

///@addtogroup FMIndex
///@{

///
/// This class represents an FM-Index, i.e. the self-compressed text index by Ferragina & Manzini.
/// An FM-Index is built on top of the following ingredients:
///     - the BWT of a text T
///     - a rank dictionary (\ref rank_dictionary) of the given BWT
///     - a sampled suffix array of T
///
/// Given the above, it allows to count and locate all the occurrences in T of arbitrary patterns
/// P in O(length(P)) time.
/// Moreover, it does so with an incremental algorithm that proceeds character by character,
/// a fundamental property that allows to implement sophisticated pattern matching algorithms
/// based on backtracking.
///
/// fm_index is <i>storage-free</i>, in the sense it doesn't directly hold any allocated
/// data - hence it can be instantiated both on host and device data-structures, and it can
/// be conveniently passed as a kernel parameter (provided its template parameters are also
/// PODs)
///
/// \tparam TRankDictionary     a rank dictionary (see \ref RankDictionaryModule)
/// \tparam TSuffixArray        a sampled suffix array context implementing the \ref SSAInterface (see \ref SSAModule)
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
struct fm_index
{
    typedef TRankDictionary                      rank_dictionary_type;
    typedef typename TRankDictionary::text_type  bwt_type;
    typedef TSuffixArray                         suffix_array_type;

    typedef typename TRankDictionary::index_type    index_type;
    typedef typename TRankDictionary::range_type    range_type;
    typedef typename TRankDictionary::vec4_type     vec4_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type      length() const { return m_length; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type      primary() const { return m_primary; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type      count(const uint32 c) const { return m_L2[c+1] - m_L2[c]; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type      L2(const uint32 c) const { return m_L2[c]; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE TRankDictionary rank_dict() const { return m_rank_dict; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE TSuffixArray    sa() const { return m_sa; }
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bwt_type        bwt() const { return m_rank_dict.text; }

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fm_index(
        const index_type      length,
        const index_type      primary,
        const index_type*     L2,
        const TRankDictionary rank_dict,
        const TSuffixArray    sa) :
        m_length( length ),
        m_primary( primary ),
        m_L2( L2 ),
        m_rank_dict( rank_dict ),
        m_sa( sa )
    {}

    index_type          m_length;
    index_type          m_primary;
    const index_type*   m_L2;
    TRankDictionary     m_rank_dict;
    TSuffixArray        m_sa;
};

/// \relates fm_index
/// return the number of occurrences of c in the range [0,k] of the given FM-index.
///
/// \param fmi      FM-index
/// \param k        range search delimiter
/// \param c        query character
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::index_type rank(
    const fm_index<TRankDictionary,TSuffixArray>&                   fmi,
    typename fm_index<TRankDictionary,TSuffixArray>::index_type     k,
    uint8                                                           c);

/// \relates fm_index
/// return the number of occurrences of c in the ranges [0,l] and [0,r] of the
/// given FM-index.
///
/// \param fmi      FM-index
/// \param range    range query [l,r]
/// \param c        query character
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type rank(
    const fm_index<TRankDictionary,TSuffixArray>&                   fmi,
    typename fm_index<TRankDictionary,TSuffixArray>::range_type     range,
    uint8                                                           c);

/// \relates fm_index
/// return the number of occurrences of all characters in the range [0,k] of the
/// given FM-index.
///
/// \param fmi      FM-index
/// \param k        range search delimiter
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::vec4_type rank4(
    const fm_index<TRankDictionary,TSuffixArray>&                   fmi,
    typename fm_index<TRankDictionary,TSuffixArray>::index_type     k);

/// \relates fm_index
/// return the number of occurrences of all characters in the range [0,k] of the
/// given FM-index.
///
/// \param fmi      FM-index
/// \param range    range query [l,r]
/// \param outl     first output
/// \param outh     second output
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void rank4(
    const fm_index<TRankDictionary,TSuffixArray>&                   fmi,
    typename fm_index<TRankDictionary,TSuffixArray>::range_type     range,
    typename fm_index<TRankDictionary,TSuffixArray>::vec4_type*     outl,
    typename fm_index<TRankDictionary,TSuffixArray>::vec4_type*     outh);

/// \relates fm_index
/// return the range of occurrences of a pattern in the given FM-index.
///
/// \param fmi          FM-index
/// \param pattern      query string
/// \param pattern_len  query string length
///
template <
    typename TRankDictionary,
    typename TSuffixArray,
    typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type match(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const Iterator                                                      pattern,
    const uint32                                                        pattern_len);

/// \relates fm_index
/// return the range of occurrences of a pattern in the given FM-index.
///
/// \param fmi          FM-index
/// \param pattern      query string
/// \param pattern_len  query string length
/// \param range        start range
///
template <
    typename TRankDictionary,
    typename TSuffixArray,
    typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type match(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const Iterator                                                      pattern,
    const uint32                                                        pattern_len,
    const typename fm_index<TRankDictionary,TSuffixArray>::range_type   range);

/// \relates fm_index
/// return the range of occurrences of a reversed pattern in the given FM-index.
///
/// \param fmi          FM-index
/// \param pattern      query string
/// \param pattern_len  query string length
///
template <
    typename TRankDictionary,
    typename TSuffixArray,
    typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type match_reverse(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const Iterator                                                      pattern,
    const uint32                                                        pattern_len);

// \relates fm_index
// computes the inverse psi function at a given index, without using the reduced SA
//
// \param fmi          FM-index
// \param i            query index
// \return             base inverse psi function value and offset
//
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::index_type basic_inv_psi(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const typename fm_index<TRankDictionary,TSuffixArray>::range_type   i);

/// \relates fm_index
/// computes the inverse psi function at a given index
///
/// \param fmi          FM-index
/// \param i            query index
/// \return             base inverse psi function value and offset
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type inv_psi(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const typename fm_index<TRankDictionary,TSuffixArray>::index_type   i);

/// \relates fm_index
/// return the linear coordinate of the suffix that prefixes the i-th row of the BWT matrix.
///
/// \param fmi          FM-index
/// \param i            query index
/// \return             position of the suffix that prefixes the query index
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::index_type locate(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const typename fm_index<TRankDictionary,TSuffixArray>::index_type   i);

/// \relates fm_index
/// return the position of the suffix that prefixes the i-th row of the BWT matrix in the sampled SA,
/// and its relative offset
///
/// \param fmi          FM-index
/// \param i            query index
/// \return             a pair formed by the position of the suffix that prefixes the query index
///                     in the sampled SA and its relative offset
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::range_type locate_ssa_iterator(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const typename fm_index<TRankDictionary,TSuffixArray>::index_type   i);

/// \relates fm_index
/// return the position of the suffix that prefixes the i-th row of the BWT matrix.
///
/// \param fmi          FM-index
/// \param iter         iterator to the sampled SA
/// \return             final linear coordinate
///
template <
    typename TRankDictionary,
    typename TSuffixArray>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fm_index<TRankDictionary,TSuffixArray>::index_type lookup_ssa_iterator(
    const fm_index<TRankDictionary,TSuffixArray>&                       fmi,
    const typename fm_index<TRankDictionary,TSuffixArray>::range_type   it);

#ifdef __CUDACC__
#if defined(MOD_NAMESPACE)
MOD_NAMESPACE_BEGIN
#endif

#if USE_TEX
texture<uint32> s_count_table_tex;
#endif
///
/// Helper structure for handling the (global) count table texture
///
struct count_table_texture
{
    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE uint32 operator[] (const uint32 i) const;

    /// bind texture
    ///
    NVBIO_FORCEINLINE NVBIO_HOST static void bind(const uint32* count_table);

    /// unbind texture
    ///
    NVBIO_FORCEINLINE NVBIO_HOST static void unbind();
};

#if defined(MOD_NAMESPACE)
MOD_NAMESPACE_END
#endif
#endif

///@} FMIndex

} // namespace nvbio

#include <nvbio/fmindex/fmindex_inl.h>
