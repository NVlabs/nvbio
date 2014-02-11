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

#include <nvbio/io/output/output_stats.h>
#include <nvbio/io/output/output_types.h>

namespace nvbio {
namespace io {

// keep track of alignment statistics for a given alignment
// xxxnsubtil: it's unclear to me whether the original code expected the first mate
// (i.e., the 'alignment' argument) to always be the anchor and the second to always
// be the opposite mate
void IOStats::track_alignment_statistics(const AlignmentData& alignment,
                                         const AlignmentData& mate,
                                         const uint8 mapq)
{
    n_reads++;

    // keep track of mapping quality histogram
    mapq_bins[mapq]++;

    if (!alignment.best->is_aligned())
    {
        // sanity check
        NVBIO_CUDA_ASSERT(alignment.second_best->is_mapped() == false);

        // if first anchor was not mapped, nothing was; count as unmapped
        mapped_ed_correlation[0][0]++;
        return;
    }

    // count this read as mapped
    n_mapped++;

    if (!alignment.second_best->is_aligned())
    {
        // we only have one score; count as a unique alignment
        n_unique++;
    } else {
        // we have two best scores, which implies two (or more) alignment positions
        // count as multiple alignment
        n_multiple++;

        // compute final alignment score
        int32 first  = alignment.best->score();
        int32 second = alignment.second_best->score();

        if (mate.valid)
        {
            first += mate.best->score();
            second += mate.second_best->score();
        }

        // if the two best scores are equal, count as ambiguous
        if (first == second)
            n_ambiguous++;
        else {
            // else, the first score must be higher...
            NVBIO_CUDA_ASSERT(first > second);
            /// ... which counts as a nonambiguous alignment
            n_unambiguous++;
        }

        // compute edit distance scores
        uint32 first_ed  = alignment.best->ed();
        uint32 second_ed = alignment.second_best->score();

        // update best edit-distance histograms
        if (first_ed < mapped_ed_histogram.size())
        {
            mapped_ed_histogram[first_ed]++;
            if (alignment.best->m_rc)
            {
                mapped_ed_histogram_fwd[first_ed]++;
            } else {
                mapped_ed_histogram_rev[first_ed]++;
            }
        }

        // track edit-distance correlation
        if (first_ed + 1 < 64)
        {
            if (second_ed + 1 < 64)
            {
                mapped_ed_correlation[first_ed + 1][second_ed + 1]++;
            } else {
                mapped_ed_correlation[first_ed + 1][0]++;
            }
        }
    }
}

}
}
