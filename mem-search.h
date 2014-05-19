/*
 * Copyright (c) 2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#include <nvbio/io/sequence/sequence.h>

/// initialize the MEM-search pipeline
///
void mem_init(struct pipeline_state *pipeline);

/// search MEMs for the given batch of reads
///
void mem_search(struct pipeline_state *pipeline, const nvbio::io::SequenceDataDevice<nvbio::DNA_N> *batch);

/// given the first read in a chunk, determine a suitably sized chunk of reads
/// (for which we can locate all MEMs in one go), updating pipeline::chunk
///
void fit_read_chunk(
    struct pipeline_state                               *pipeline,
    const nvbio::io::SequenceDataDevice<nvbio::DNA_N>   *batch,
    const nvbio::uint32                                 read_begin);    // first read in the chunk

/// locate all mems in the range defined by pipeline::chunk
///
void mem_locate(struct pipeline_state *pipeline, const nvbio::io::SequenceDataDevice<nvbio::DNA_N> *batch);
