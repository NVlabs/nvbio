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
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/stats.h>
#include <nvbio/basic/threads.h>
#include <nvbio/basic/timer.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

//
// A class implementing a background input thread, providing
// a set of input read-streams which are read in parallel to the
// operations performed by the main thread.
//

struct InputThread : public Thread<InputThread>
{
    static const uint32 BUFFERS = 4;
    static const uint32 INVALID = 1u;

    InputThread(io::ReadDataStream* read_data_stream, Stats& _stats, const uint32 batch_size) :
        m_read_data_stream( read_data_stream ), m_stats( _stats ), m_batch_size( batch_size ), m_set(0)
    {
        for (uint32 i = 0; i < BUFFERS; ++i)
            read_data[i] = NULL;
    }

    void run();

    io::ReadDataStream* m_read_data_stream;
    Stats&              m_stats;
    uint32              m_batch_size;
    volatile uint32     m_set;

    io::ReadData* volatile read_data[BUFFERS];
};

//
// A class implementing a background input thread, providing
// a set of input read-streams which are read in parallel to the
// operations performed by the main thread.
//

struct InputThreadPaired : public Thread<InputThreadPaired>
{
    static const uint32 BUFFERS = 4;
    static const uint32 INVALID = 1u;

    InputThreadPaired(io::ReadDataStream* read_data_stream1, io::ReadDataStream* read_data_stream2, Stats& _stats, const uint32 batch_size) :
        m_read_data_stream1( read_data_stream1 ), m_read_data_stream2( read_data_stream2 ), m_stats( _stats ), m_batch_size( batch_size ), m_set(0)
    {
        for (uint32 i = 0; i < BUFFERS; ++i)
            read_data1[i] = read_data2[i] = NULL;
    }

    void run();

    io::ReadDataStream* m_read_data_stream1;
    io::ReadDataStream* m_read_data_stream2;
    Stats&              m_stats;
    uint32              m_batch_size;
    volatile uint32     m_set;

    io::ReadData* volatile read_data1[BUFFERS];
    io::ReadData* volatile read_data2[BUFFERS];
};

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
