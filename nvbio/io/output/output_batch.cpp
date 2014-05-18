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

#include <nvbio/io/output/output_batch.h>
#include <nvbio/io/fmi.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/vector.h>

#include <stdio.h>
#include <stdarg.h>

namespace nvbio {
namespace io {

AlignmentResult::AlignmentResult()
{
    best[MATE_1] = Alignment::invalid();
    best[MATE_2] = Alignment::invalid();
    second_best[MATE_1] = Alignment::invalid();
    second_best[MATE_2] = Alignment::invalid();

    is_paired_end = false;
}
// copy scoring data to host, converting to io::AlignmentResult
void GPUOutputBatch::readback_scores(thrust::host_vector<io::AlignmentResult>& output,
                                     const AlignmentMate mate,
                                     const AlignmentScore score) const
{
    // copy alignment data into a staging buffer
    thrust::host_vector<io::BestAlignments> best_data_staging;
    nvbio::cuda::thrust_copy_vector(best_data_staging, best_data_dvec, count);

    // convert the contents of the staging buffer into io::AlignmentResult
    output.resize(count);
    for(uint32 c = 0; c < count; c++)
    {
        io::BestAlignments&      old_best_aln = best_data_staging[c];
        io::Alignment&           old_aln = (score == BEST_SCORE ? old_best_aln.m_a1 : old_best_aln.m_a2);
        io::AlignmentResult&    new_aln = output[c];

        if (score == BEST_SCORE)
        {
            new_aln.best[mate] = old_aln;
        } else {
            new_aln.second_best[mate] = old_aln;
        }

        if (mate == MATE_2)
        {
            new_aln.is_paired_end = true;
        }
    }
}

// copy CIGARs into host memory
void GPUOutputBatch::readback_cigars(io::HostCigarArray& host_cigar) const
{
    host_cigar.array = cigar.array;
    nvbio::cuda::thrust_copy_vector(host_cigar.coords, cigar.coords);
}

// copy MD strings back to the host
void GPUOutputBatch::readback_mds(nvbio::HostVectorArray<uint8>& host_mds) const
{
    host_mds = mds;
}

// extract alignment data for a given mate
// note that the mates can be different for the cigar, since mate 1 is always the anchor mate for cigars
AlignmentData CPUOutputBatch::get_mate(uint32 read_id, AlignmentMate mate, AlignmentMate cigar_mate)
{
    return AlignmentData(&best_alignments[read_id].best[mate],
                         &best_alignments[read_id].second_best[mate],
                         read_id,
                         read_data[mate],
                         &cigar[cigar_mate],
                         &mds[cigar_mate]);
}

// extract alignment data for the anchor mate
AlignmentData CPUOutputBatch::get_anchor(uint32 read_id)
{
    if (best_alignments[read_id].best[MATE_1].mate() == 0)
    {
        // mate 1 is the anchor
        return get_mate(read_id, MATE_1, MATE_1);
    } else {
        // mate 2 is the anchor
        return get_mate(read_id, MATE_2, MATE_1);
    }
}

// extract alignment data for the opposite mate
AlignmentData CPUOutputBatch::get_opposite_mate(uint32 read_id)
{
    if (best_alignments[read_id].best[MATE_1].mate() == 0)
    {
        // mate 2 is the opposite mate
        return get_mate(read_id, MATE_2, MATE_2);
    } else {
        // mate 1 is the opposite mate
        return get_mate(read_id, MATE_1, MATE_2);
    }
}

} // namespace io
} // namespace nvbio
