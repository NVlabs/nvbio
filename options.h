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

#include <nvbio/basic/types.h>

using namespace nvbio;

struct runtime_options
{
    // genome file name or shared memory handle name
    const char *genome_file_name;
    // input reads file name
    const char *input_file_name;
    // output alignment file name
    const char *output_file_name;

    // whether to allow using mmap() to load the genome
    bool genome_use_mmap;
    // input read batch size
    uint64 batch_size;

    // MEM search options
    uint32 min_intv;        // min and max interval sizes for MEM search
    uint32 max_intv;
    uint32 min_seed_len;    // minimum read span, MEMs that span less than this many bps will be dropped
    uint32 mems_batch;      // number of MEMs to locate at once
    uint32 w;
    uint32 max_chain_gap;
    uint32 split_len;
    uint32 split_width;
    float  mask_level;
    float  chain_drop_ratio;

    runtime_options()
    {
        genome_file_name = NULL;
        input_file_name  = NULL;
        output_file_name = NULL;

        // default options
        genome_use_mmap     = true;
        batch_size          = 256 * 1024;
        min_intv            = 1;
        max_intv            = 10000;
        min_seed_len        = 19;
        mems_batch          = 16 * 1024 * 1024;
        split_len           = 28;
        split_width         = 10;
        mask_level          = 0.5f;
        chain_drop_ratio    = 0.5f;
        w                   = 100;
        max_chain_gap       = 10000;
    };
};

extern struct runtime_options command_line_options;

void parse_command_line(int argc, char **argv);
