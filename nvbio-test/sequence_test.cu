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

// alignment_test.cu
//

#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/dna.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_mmap.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvbio;

namespace nvbio {

int sequence_test(int argc, char* argv[])
{
    char* index_name = NULL;
    char* reads_name = NULL;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-map" ) == 0)
            index_name = argv[++i];
        else if (strcmp( argv[i], "-reads" ) == 0)
            reads_name = argv[++i];
    }

    log_info(stderr,"testing sequence-data... started\n");

    if (index_name != NULL)
    {
        io::SequenceDataMMAPServer server;
        if (server.load( DNA, index_name, "test", io::SequenceFlags( io::SEQUENCE_DATA | io::SEQUENCE_NAMES ) ) == false)
        {
            log_error(stderr,"  server mapping of file %s failed\n", index_name);
            return 0;
        }

        io::SequenceDataMMAP client;
        if (client.load( "test" ) == false)
        {
            log_error(stderr,"  client mapping of file %s failed\n", index_name);
            return 0;
        }

        log_verbose(stderr, "  sequences : %u\n", client.size() );
        log_verbose(stderr, "  bps       : %u\n", client.bps() );
    }
    if (reads_name != NULL)
    {
        SharedPointer<io::SequenceDataStream> read_file( io::open_sequence_file( reads_name ) );
        if (read_file == NULL || read_file->is_ok() == false)
        {
            log_error(stderr,"  failed opening reads file %s\n", reads_name);
            return 0;
        }

        io::SequenceDataHost read_data;

        io::next( DNA_N, &read_data, read_file.get(), 10000 );

        log_verbose(stderr, "  sequences : %u\n", read_data.size() );
        log_verbose(stderr, "  bps       : %u\n", read_data.bps() );
        log_verbose(stderr, "  avg bps   : %u (min: %u, max: %u)\n",
            read_data.avg_sequence_len(),
            read_data.min_sequence_len(),
            read_data.max_sequence_len() );
    }

    log_info(stderr,"testing sequence-data... done\n");
    return 1;
}

} // namespace nvbio
