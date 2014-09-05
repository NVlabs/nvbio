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

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/scoring_queues.h>
#include <nvbio/basic/types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <string>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

struct SeedHit;
struct HitQueue;
struct SeedHitDequeArray;

// clear the persisting files
//
void persist_clear(const std::string& file_name);

// persist a set of hits
//
void persist_hits(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    count,
    const SeedHitDequeArray&                        hit_deques);

// persist a set of reads
//
void persist_reads(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    count,
    const thrust::device_vector<uint32>::iterator   iterator);

// persist a set of selected hits
//
void persist_selection(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    read_count,
    const packed_read*                              read_infos_dptr,
    const uint32                                    n_multi,
    const uint32                                    hits_queue_size,
    const ReadHitsIndex&                            hits_index,
    const HitQueues&                                hits_queue);

// persist a set of scores
//
void persist_scores(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    read_count,
    const uint32                                    n_multi,
    const uint32                                    hits_queue_size,
    const ScoringQueues&                            scoring_queues);

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
