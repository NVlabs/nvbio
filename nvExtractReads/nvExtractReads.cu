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

// nvExtractReads.cu
//

#include <nvbio/basic/timer.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/basic/dna.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

using namespace nvbio;

bool read(const char* reads_name, FILE* output_file, const io::QualityEncoding qencoding, const io::ReadEncoding flags)
{
    log_visible(stderr, "opening read file \"%s\"\n", reads_name);
    SharedPointer<nvbio::io::ReadDataStream> read_data_file(
        nvbio::io::open_read_file(reads_name,
        qencoding,
        uint32(-1),
        uint32(-1),
        flags )
    );

    if (read_data_file == NULL || read_data_file->is_ok() == false)
    {
        log_error(stderr, "    failed opening file \"%s\"\n", reads_name);
        return false;
    }

    const uint32 batch_size = 512*1024;

    std::vector<char> char_read( 1024*1024 );

    uint32 n_reads = 0;

    // loop through all read batches
    while (1)
    {
        // load a new batch of reads
        SharedPointer<io::ReadData> h_read_data( read_data_file->next( batch_size ) );
        if (h_read_data == NULL)
            break;

        // loop through all reads
        for (uint32 i = 0; i < h_read_data->size(); ++i)
        {
            const io::ReadData::read_string read = h_read_data->get_read(i);

            dna_to_string( read, read.length(), &char_read[0] );

            char_read[ read.length() ] = '\n';

            fwrite( &char_read[0], sizeof(char), read.length()+1, output_file );
        }

        n_reads += h_read_data->size();

        log_verbose(stderr,"\r    %u reads    ", n_reads);
    }
    log_verbose_cont(stderr,"\n");
    return true;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        log_info(stderr, "nvExtractReads [options] input output\n");
        log_info(stderr, "  extract a set of reads to a plain ASCII file with one read per line (.txt)\n\n");
        log_info(stderr, "options:\n");
        log_info(stderr, "  --verbosity\n");
        log_info(stderr, "  -F | --skip-forward          skip forward strand\n");
        log_info(stderr, "  -R | --skip-reverse          skip forward strand\n");
        exit(0);
    }

    const char* reads_name  = argv[argc-2];
    const char* out_name    = argv[argc-1];
    bool  forward           = true;
    bool  reverse           = true;
    io::QualityEncoding qencoding = io::Phred33;

    for (int i = 0; i < argc - 2; ++i)
    {
        if (strcmp( argv[i], "-verbosity" ) == 0 ||
                 strcmp( argv[i], "--verbosity" ) == 0)
        {
            set_verbosity( Verbosity( atoi( argv[++i] ) ) );
        }
        else if (strcmp( argv[i], "-F" )             == 0 ||
                 strcmp( argv[i], "--skip-forward" ) == 0)  // skip forward strand
        {
            forward = false;
        }
        else if (strcmp( argv[i], "-R" ) == 0 ||
                 strcmp( argv[i], "--skip-reverse" ) == 0)  // skip reverse strand
        {
            reverse = false;
        }
    }

    FILE* output_file = fopen( out_name, "w" );
    if (output_file == NULL)
    {
        log_error(stderr, "    failed opening file \"%s\"\n", out_name);
        return 1;
    }

    log_visible(stderr,"nvExtractReads... started\n");

    uint32       encoding_flags  = 0u;
    if (forward) encoding_flags |= io::FORWARD;
    if (reverse) encoding_flags |= io::REVERSE_COMPLEMENT;

    if (read( reads_name, output_file, qencoding, io::ReadEncoding(encoding_flags) ) == false)
        return 1;

    fclose( output_file );

    log_visible(stderr,"nvExtractReads... done\n");
    return 0;
}
