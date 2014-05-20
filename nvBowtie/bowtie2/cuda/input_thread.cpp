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

#include <nvBowtie/bowtie2/cuda/input_thread.h>
#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvbio/io/alignments.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/stats.h>
#include <nvbio/io/output/output_utils.h>
#include <nvbio/basic/threads.h>
#include <nvbio/basic/timer.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

void InputThread::run()
{
    log_verbose( stderr, "starting background input thread\n" );

    while (1u)
    {
        // poll until the set is done reading & ready to be reused
        while (read_data[m_set] != NULL) {}

        //// lock the set to flush
        //ScopedLock lock( &m_lock[m_set] );

        Timer timer;
        timer.start();

        const int ret = io::next( DNA_N, &read_data_storage[ m_set ], m_read_data_stream, m_batch_size );

        timer.stop();

        if (ret)
        {
            m_stats.read_io.add( read_data_storage[ m_set ].size(), timer.seconds() );

            // mark the set as done
            read_data[ m_set ] = &read_data_storage[ m_set ];
        }
        else
        {
            // mark this as an invalid entry
            read_data[ m_set ] = (io::SequenceDataHost*)INVALID;
            break;
        }

        // switch to the next set
        m_set = (m_set + 1) % BUFFERS;
    }
}

void InputThreadPaired::run()
{
    log_verbose( stderr, "starting background paired-end input thread\n" );

    while (1u)
    {
        // poll until the set is done reading & ready to be reused
        while (read_data1[m_set] != NULL || read_data2[m_set] != NULL) {}

        //// lock the set to flush
        //ScopedLock lock( &m_lock[m_set] );

        Timer timer;
        timer.start();

        const int ret1 = io::next( DNA_N, &read_data_storage1[ m_set ], m_read_data_stream1, m_batch_size );
        const int ret2 = io::next( DNA_N, &read_data_storage2[ m_set ], m_read_data_stream2, m_batch_size );

        timer.stop();

        if (ret1 && ret2)
        {
            m_stats.read_io.add( read_data_storage1[ m_set ].size(), timer.seconds() );

            // mark the set as done
            read_data1[ m_set ] = &read_data_storage1[ m_set ];
            read_data2[ m_set ] = &read_data_storage2[ m_set ];
        }
        else
        {
            // mark this as an invalid entry
            read_data1[ m_set ] = (io::SequenceDataHost*)INVALID;
            read_data2[ m_set ] = (io::SequenceDataHost*)INVALID;
            break;
        }

        // switch to the next set
        m_set = (m_set + 1) % BUFFERS;
    }
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
