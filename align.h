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

/// initialize the alignment pipeline
///
void align_init(struct pipeline_state *pipeline, const nvbio::io::SequenceDataDevice<nvbio::DNA_N> *batch);

/// perform banded alignment
///
/// \return     the number of remaining active reads to align
///
nvbio::uint32 align_short(struct pipeline_state *pipeline, const nvbio::io::SequenceDataDevice<nvbio::DNA_N> *batch);
