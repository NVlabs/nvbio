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

#include <nvbio/io/output/output_types.h>
#include <nvbio/io/output/output_stats.h>
#include <nvbio/io/output/output_file.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <nvbio/basic/vector_array.h>
#include <nvbio/io/sequence/sequence.h>

#include <stdio.h>

namespace nvbio {
namespace io {

// a batch of alignment results on the CPU
struct CPUOutputBatch
{
    uint32 count;

    thrust::host_vector<io::AlignmentResult> best_alignments;

    // we have two cigar and MDS arrays, one for each mate
    HostCigarArray                               cigar[2];
    HostMdsArray                                 mds[2];

    // pointer to the host-side read data for each mate
    const io::SequenceDataHost*                  read_data[2];

    CPUOutputBatch()
        : count(0)
    { }

    // extract alignment data for a given mate
    AlignmentData get_mate(uint32 read_id, AlignmentMate mate, AlignmentMate cigar_mate);
    // extract alignment data for the anchor mate
    AlignmentData get_anchor(uint32 read_id);
    // extract alignment data for the opposite mate
    AlignmentData get_opposite_mate(uint32 read_id);
};

// base class for representing a batch of alignment results on the device
struct GPUOutputBatch
{
public:
    uint32                                     count;

    thrust::device_vector<io::BestAlignments>& best_data_dvec;
    DeviceCigarArray                           cigar;
    nvbio::DeviceVectorArray<uint8>&           mds;

    const io::SequenceDataDevice&              read_data;

    GPUOutputBatch(uint32                                         _count,
                   thrust::device_vector<io::BestAlignments>&     _best_data_dvec,
                   DeviceCigarArray                               _cigar,
                   nvbio::DeviceVectorArray<uint8>&               _mds,
                   const io::SequenceDataDevice&                  _read_data)
            : count(_count),
              best_data_dvec(_best_data_dvec),
              cigar(_cigar),
              mds(_mds),
              read_data(_read_data)
    { }

    // copy best score data into host memory
    void readback_scores(thrust::host_vector<io::AlignmentResult>& output,
                         const AlignmentMate mate,
                         const AlignmentScore score) const;
    // copy cigars into host memory
    void readback_cigars(HostCigarArray& host_cigars) const;
    // copy md strings into host memory
    void readback_mds(nvbio::HostVectorArray<uint8>& host_mds) const;
};

} // namespace io
} // namespace nvbio
