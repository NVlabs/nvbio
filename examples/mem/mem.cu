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

// mem.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/basic/dna.h>
#include <nvbio/strings/string_set.h>
#include <nvbio/strings/infix.h>
#include <nvbio/strings/seeds.h>
#include <nvbio/fmindex/mem.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/fmi.h>

using namespace nvbio;

// main test entry point
//
int main(int argc, char* argv[])
{
    //
    // perform some basic option parsing
    //

    const uint32 batch_reads   =   1*1024*1024;
    const uint32 batch_bps     = 100*1024*1024;

    const char* reads = argv[argc-1];
    const char* index = argv[argc-2];

    uint32 max_reads        = uint32(-1);
    uint32 min_intv         = 1u;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-max-reads" ) == 0)
            max_reads = uint32( atoi( argv[++i] ) );
        else if (strcmp( argv[i], "-min-intv" ) == 0)
            min_intv = int16( atoi( argv[++i] ) );
    }

    const uint32 fm_flags = io::FMIndexData::GENOME  |
                            io::FMIndexData::FORWARD |
                            io::FMIndexData::REVERSE |
                            io::FMIndexData::SA;

    io::FMIndexData *h_fmi = NULL;
    io::FMIndexDataMMAP mmap_loader;
    io::FMIndexDataRAM file_loader;

    if (mmap_loader.load( index ))
    {
        h_fmi = &mmap_loader;
    } else {
        if (!file_loader.load( index, fm_flags ))
        {
            log_error(stderr, "    failed loading index \"%s\"\n", index);
            return 1u;
        }

        h_fmi = &file_loader;
    }

    // build its device version
    const io::FMIndexDataDevice d_fmi( *h_fmi, fm_flags );

    typedef io::FMIndexDataDevice::stream_type genome_type;

    // fetch the genome string
    const uint32      genome_len = d_fmi.genome_length();
    const genome_type d_genome( d_fmi.genome_stream() );

    // open a read file
    log_info(stderr, "  opening reads file... started\n");

    SharedPointer<io::ReadDataStream> read_data_file(
        io::open_read_file(
            reads,
            io::Phred33,
            2*max_reads,
            uint32(-1),
            io::ReadEncoding( io::FORWARD | io::REVERSE_COMPLEMENT ) ) );

    // check whether the file opened correctly
    if (read_data_file == NULL || read_data_file->is_ok() == false)
    {
        log_error(stderr, "    failed opening file \"%s\"\n", reads);
        return 1u;
    }
    log_info(stderr, "  opening reads file... done\n");

    typedef io::FMIndexDataDevice::fm_index_type        fm_index_type;
    typedef MEMFilterDevice<fm_index_type>              mem_filter_type;

    // fetch the FM-index
    const fm_index_type f_index = d_fmi.index();
    const fm_index_type r_index = d_fmi.rindex();

    // create a MEM filter
    mem_filter_type mem_filter;

    const uint32 mems_batch = 16*1024*1024;
    nvbio::vector<device_tag,mem_filter_type::mem_type> mems( mems_batch );

    while (1)
    {
        // load a batch of reads
        SharedPointer<io::ReadData> h_read_data( read_data_file->next( batch_reads, batch_bps ) );
        if (h_read_data == NULL)
            break;

        log_info(stderr, "  loading reads... started\n");

        // copy it to the device
        const io::ReadDataDevice d_read_data( *h_read_data );

        const uint32 n_reads = d_read_data.size() / 2;

        log_info(stderr, "  loading reads... done\n");
        log_info(stderr, "    %u reads\n", n_reads);

        log_info(stderr, "  ranking MEMs... started\n");

        Timer timer;
        timer.start();

        mem_filter.rank(
            f_index,
            r_index,
            d_read_data.const_read_string_set(),
            min_intv );

        cudaDeviceSynchronize();
        timer.stop();

        const uint64 n_mems = mem_filter.n_mems();

        log_info(stderr, "  ranking MEMs... done\n");
        log_info(stderr, "    %.1f avg ranges\n", float( mem_filter.n_ranges() ) / float( n_reads ) );
        log_info(stderr, "    %.1f avg MEMs\n", float( n_mems ) / float( n_reads ) );
        log_info(stderr, "    %.1f K reads/s\n", 1.0e-3f * float(n_reads) / timer.seconds());

        log_info(stderr, "  locating MEMs... started\n");

        float locate_time = 0.0f;

        // loop through large batches of hits and locate & merge them
        for (uint64 mems_begin = 0; mems_begin < n_mems; mems_begin += mems_batch)
        {
            const uint64 mems_end = nvbio::min( mems_begin + mems_batch, n_mems );

            timer.start();

            mem_filter.locate(
                mems_begin,
                mems_end,
                mems.begin() );

            cudaDeviceSynchronize();
            timer.stop();
            locate_time += timer.seconds();

            log_verbose(stderr, "\r    %5.2f%% (%4.1f M MEMs/s)",
                 100.0f * float( mems_end ) / float( n_mems ),
                1.0e-6f * float( mems_end ) / locate_time );
        }

        log_info(stderr, "  locating MEMs... done\n");
    }
    return 0;
}
