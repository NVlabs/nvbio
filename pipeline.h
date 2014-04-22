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

#include <nvbio/io/output/output_file.h>

#include "mem-search.h"

using namespace nvbio;

struct pipeline_stats
{
    pipeline_stats() :
        time        ( 0.0f ),
        io_time     ( 0.0f ),
        search_time ( 0.0f ),
        locate_time ( 0.0f ),
        chain_time  ( 0.0f ),
        n_reads     ( 0 ),
        n_mems      ( 0 ),
        n_chains    ( 0 )
    {}

    float time;
    float io_time;
    float search_time;
    float locate_time;
    float chain_time;

    uint64 n_reads;
    uint64 n_mems;
    uint64 n_chains;
};

struct pipeline_context 
{
    io::OutputFile          *output;
    struct mem_state        mem;
    struct read_chunk       chunk;
    struct pipeline_stats   stats;
};
