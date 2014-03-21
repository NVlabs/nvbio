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

// nvBWT.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <crc/crc.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/bnt.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/fmindex/dna.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fasta/fasta.h>
#include <nvbio/sufsort/sufsort.h>
#include "filelist.h"


using namespace nvbio;

#define _32_32 0
#define _64_64 1
#define _32_64 2

#define SA_REP _64_64

#define DIVSUFSORT 0
#define SAIS       1
#define BWTSW      2

#define BYTE_PACKING 0
#define WORD_PACKING 1

#if (SA_REP == _32_32)
typedef uint32 SA_storage_type;
typedef uint32 SA_facade_type;
#elif (SA_REP == _64_64)
typedef uint64 SA_storage_type;
typedef uint64 SA_facade_type;
#else
typedef uint32 SA_storage_type;
typedef int64  SA_facade_type;
#endif

void bwt_bwtgen(const char *fn_pac, const char *fn_bwt);

unsigned char nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};

#ifdef WIN32
inline void  srand_bp(const unsigned int s) { srand(s); }
inline float frand() { return float(rand()) / float(RAND_MAX); }
inline uint8 rand_bp() { return uint8( frand() * 4 ); }
#else
inline void  srand_bp(const unsigned int s) { srand48(s); }
inline uint8 rand_bp() { return uint8( lrand48() & 3u ); }
#endif

struct Counter
{
    Counter() : m_size(0), m_reads(0) {}

    void begin_read() { m_reads++; }
    void end_read() {}

    void id(const uint8 c) {}
    void read(const uint8 c) { m_size++; }

    uint64 m_size;
    uint32 m_reads;
};
template <typename StorageType>
struct Writer
{
    typedef PackedStream<StorageType*,uint8,2,true,SA_facade_type> stream_type;

    Writer(StorageType* storage, const uint32 reads, const uint64 max_size) :
        m_max_size(max_size), m_size(0), m_stream( storage )
    {
        m_bntseq.seed = 11;
        m_bntseq.anns_data.resize( reads );
        m_bntseq.anns_info.resize( reads );

        srand_bp( m_bntseq.seed );
    }

    void begin_read()
    {
        BNTAnnData& ann_data = m_bntseq.anns_data[ m_bntseq.n_seqs ];
        ann_data.len    = 0;
        ann_data.gi     = 0;
        ann_data.offset = m_size;
        ann_data.n_ambs = 0;

        BNTAnnInfo& ann_info = m_bntseq.anns_info[ m_bntseq.n_seqs ];
        ann_info.anno   = "null";

        m_lasts = 0;
    }
    void end_read()
    {
        m_bntseq.n_seqs++;
    }

    void id(const uint8 c)
    {
        m_bntseq.anns_info[ m_bntseq.n_seqs ].name.push_back(char(c));
    }
    void read(const uint8 s)
    {
        if (m_size < m_max_size)
        {
            const uint8 c = nst_nt4_table[s];

            m_stream[ SA_facade_type(m_size) ] = c < 4 ? c : rand_bp();

            if (c >= 4) // we have an N
            {
                if (m_lasts == s) // contiguous N
                {
                    // increment length of the last hole
                    ++m_bntseq.ambs.back().len;
                }
                else
                {
                    // beginning of a new hole
                    BNTAmb amb;
                    amb.len    = 1;
                    amb.offset = m_size;
                    amb.amb    = s;

                    m_bntseq.ambs.push_back( amb );

                    ++m_bntseq.anns_data[ m_bntseq.n_seqs ].n_ambs;
                    ++m_bntseq.n_holes;
                }
            }
            // save last symbol
            m_lasts = s;

            // update sequence length
            BNTAnnData& ann_data = m_bntseq.anns_data[ m_bntseq.n_seqs ];
            ann_data.len++;
        }

        m_bntseq.l_pac++;

        m_size++;
    }

    uint64      m_max_size;
    uint64      m_size;
    stream_type m_stream;

    BNTSeq      m_bntseq;
    uint8       m_lasts;
};

template <typename StreamType>
bool save_stream(FILE* output_file, const uint64 seq_words, const StreamType* stream)
{
    for (uint64 words = 0; words < seq_words; words += 1024)
    {
        const uint32 n_words = (uint32)nvbio::min( uint64(1024u), uint64(seq_words - words) );
        if (fwrite( stream + words, sizeof(StreamType), n_words, output_file ) != n_words)
            return false;
    }
    return true;
}

int build(
    const char*  input_name,
    const char*  output_name,
    const char*  pac_name,
    const char*  rpac_name,
    const char*  bwt_name,
    const char*  rbwt_name,
    const uint64 max_length)
{
    std::vector<std::string> sortednames;
    list_files(input_name, sortednames);

    uint32 n_inputs = (uint32)sortednames.size();
    log_info(stderr, "\ncounting bps... started\n");
    // count entire sequence length
    Counter counter;

    for (uint32 i = 0; i < n_inputs; ++i)
    {
        log_info(stderr, "  counting \"%s\"\n", sortednames[i].c_str());

        FASTA_inc_reader fasta( sortednames[i].c_str() );
        if (fasta.valid() == false)
        {
            log_error(stderr, "  unable to open file\n");
            exit(1);
        }

        while (fasta.read( 1024, counter ) == 1024);
    }
    log_info(stderr, "counting bps... done\n");

    const uint64 seq_length   = nvbio::min( (uint64)counter.m_size, (uint64)max_length );
    const uint32 bps_per_word = sizeof(uint32)*4u;
    const uint64 seq_words    = (seq_length + bps_per_word - 1u) / bps_per_word;

    log_info(stderr, "\nstats:\n");
    log_info(stderr, "  reads           : %u\n", counter.m_reads );
    log_info(stderr, "  sequence length : %llu bps (%.1f MB)\n",
        seq_length,
        float(seq_words*sizeof(uint32))/float(1024*1024));
    log_info(stderr, "  buffer size     : %.1f MB\n",
        2*seq_words*sizeof(uint32)/1.0e6f );

    // allocate the actual storage
    thrust::host_vector<uint32> h_base_storage( seq_words );
    thrust::host_vector<uint32> h_bwt_storage( seq_words );

    uint32* h_base_stream = nvbio::plain_view( h_base_storage );
    uint32* h_bwt_stream  = nvbio::plain_view( h_bwt_storage );

    typedef PackedStream<const uint32*,uint8,2,true,SA_facade_type> const_stream_type;
    typedef PackedStream<      uint32*,uint8,2,true,SA_facade_type>       stream_type;

    stream_type h_stream( h_base_stream );
    stream_type h_bwt( h_bwt_stream );

    log_info(stderr, "\nbuffering bps... started\n");
    // read all files
    {
        Writer<uint32> writer( h_base_stream, counter.m_reads, seq_length );

        for (uint32 i = 0; i < n_inputs; ++i)
        {
            log_info(stderr, "  buffering \"%s\"\n", sortednames[i].c_str());

            FASTA_inc_reader fasta( sortednames[i].c_str() );
            if (fasta.valid() == false)
            {
                log_error(stderr, "  unable to open file!\n");
                exit(1);
            }

            while (fasta.read( 1024, writer ) == 1024);
        }

        save_bns( writer.m_bntseq, output_name );
    }
    log_info(stderr, "buffering bps... done\n");
    {
        const uint32 crc = crcCalc( h_stream.begin(), uint32(seq_length) );
        log_info(stderr, "  crc: %u\n", crc);
    }

    // writing
    if (pac_name)
    {
        log_info(stderr, "\nwriting \"%s\"... started\n", pac_name);

        const uint32 bps_per_byte = 4u;
        const uint64 seq_bytes    = (seq_length + bps_per_byte - 1u) / bps_per_byte;

        //
        // .pac file
        //

        FILE* output_file = fopen( pac_name, "wb" );
        if (output_file == NULL)
        {
            log_error(stderr, "  could not open output file \"%s\"!\n", pac_name );
            exit(1);
        }

        if (save_stream( output_file, seq_bytes, (uint8*)h_base_stream ) == false)
        {
            log_error(stderr, "  writing failed!\n");
            exit(1);
        }
		// the following code makes the pac file size always (l_pac/4+1+1)
        if (seq_length % 4 == 0)
        {
		    const uint8 ct = 0;
		    fwrite( &ct, 1, 1, output_file );
        }
        {
            const uint8 ct = seq_length % 4;
	        fwrite( &ct, 1, 1, output_file );
        }

        fclose( output_file );

        //
        // .rpac file
        //

        output_file = fopen( rpac_name, "wb" );
        if (output_file == NULL)
        {
            log_error(stderr, "  could not open output file \"%s\"!\n", rpac_name );
            exit(1);
        }

        // reuse the bwt storage to build the reverse
        uint32* h_rbase_stream = h_bwt_stream;
        stream_type h_rstream( h_rbase_stream );

        // reverse the string
        for (uint32 i = 0; i < seq_length; ++i)
            h_rstream[i] = h_stream[ seq_length - i - 1u ];

        if (save_stream( output_file, seq_bytes, (uint8*)h_rbase_stream ) == false)
        {
            log_error(stderr, "  writing failed!\n");
            exit(1);
        }
		// the following code makes the pac file size always (l_pac/4+1+1)
        if (seq_length % 4 == 0)
        {
		    const uint8 ct = 0;
		    fwrite( &ct, 1, 1, output_file );
        }
        {
            const uint8 ct = seq_length % 4;
	        fwrite( &ct, 1, 1, output_file );
        }

        fclose( output_file );

        log_info(stderr, "writing \"%s\"... done\n", pac_name);
    }

    try
    {
        BWTParams params;
        uint32    primary;

        thrust::device_vector<uint32> d_base_storage( h_base_storage );
        thrust::device_vector<uint32> d_bwt_storage( seq_words );

        const_stream_type d_string( nvbio::plain_view( d_base_storage ) );
              stream_type d_bwt( nvbio::plain_view( d_bwt_storage ) );

        Timer timer;

        log_info(stderr, "\nbuilding forward BWT... started\n");
        timer.start();

        primary = cuda::bwt(
            seq_length,
            d_string.begin(),
            d_bwt.begin(),
            &params );

        timer.stop();
        log_info(stderr, "building forward BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);

        // save it to disk
        {
            // copy to the host
            h_bwt_storage = d_bwt_storage;

            const uint32 cumFreq[4] = { 0, 0, 0, 0 };
            log_info(stderr, "\nwriting \"%s\"... started\n", bwt_name);
            FILE* output_file = fopen( bwt_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", bwt_name );
                exit(1);
            }
            fwrite( &primary, sizeof(uint32), 1, output_file );
            fwrite( cumFreq,  sizeof(uint32), 4, output_file );
            if (save_stream( output_file, seq_words, h_bwt_stream ) == false)
            {
                log_error(stderr, "  writing failed!\n");
                exit(1);
            }
            log_info(stderr, "writing \"%s\"... done\n", bwt_name);
        }

        // reverse the string in h_base_storage
        {
            // reuse the bwt storage to build the reverse
            uint32* h_rbase_stream = h_bwt_stream;
            stream_type h_rstream( h_rbase_stream );

            // reverse the string
            for (uint32 i = 0; i < seq_length; ++i)
                h_rstream[i] = h_stream[ seq_length - i - 1u ];

            // and now swap the vectors
            h_bwt_storage.swap( h_base_storage );
            std::swap( h_base_stream, h_bwt_stream );

            // and copy back the new string to the device
            d_base_storage = h_base_storage;
        }

        log_info(stderr, "\nbuilding reverse BWT... started\n");
        timer.start();

        primary = cuda::bwt(
            seq_length,
            d_string.begin(),
            d_bwt.begin(),
            &params );

        timer.stop();
        log_info(stderr, "building reverse BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);

        // save it to disk
        {
            // copy to the host
            h_bwt_storage = d_bwt_storage;

            const uint32 cumFreq[4] = { 0, 0, 0, 0 };
            log_info(stderr, "\nwriting \"%s\"... started\n", rbwt_name);
            FILE* output_file = fopen( rbwt_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", rbwt_name );
                exit(1);
            }
            fwrite( &primary, sizeof(uint32), 1, output_file );
            fwrite( cumFreq,  sizeof(uint32), 4, output_file );
            fclose( output_file );
            if (save_stream( output_file, seq_words, h_bwt_stream ) == false)
            {
                log_error(stderr, "  writing failed!\n");
                exit(1);
            }
            log_info(stderr, "writing \"%s\"... done\n", rbwt_name);
        }
    }
    catch (...)
    {
        log_info(stderr,"error: unknown exception!\n");
        exit(1);
    }
    return 0;
}

int main(int argc, char* argv[])
{
    crcInit();

    if (argc < 2)
    {
        log_info(stderr, "please specify input and output file names, e.g:\n");
        log_info(stderr, "  nvBWT [options] myinput.*.fa output-prefix\n");
        log_info(stderr, "  options:\n");
        log_info(stderr, "    -m     max_length\n");
    }
    log_info(stderr, "arch       : %lu bit\n", sizeof(void*)*8u);
    log_info(stderr, "SA storage : %lu bits\n", sizeof(SA_storage_type)*8u);
    log_info(stderr, "SA facade  : %lu bits\n", sizeof(SA_facade_type)*8u);

    const char* file_names[2] = { NULL, NULL };
    uint64 max_length = uint64(-1);

    uint32 n_files = 0;
    for (int32 i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp( arg, "-m" ) == 0)
        {
            max_length = atoi( argv[i+1] );
            ++i;
        }
        else
            file_names[ n_files++ ] = argv[i];
    }

    const char* input_name  = file_names[0];
    const char* output_name = file_names[1];
    std::string pac_string  = std::string( output_name ) + ".pac";
    const char* pac_name    = pac_string.c_str();
    std::string rpac_string = std::string( output_name ) + ".rpac";
    const char* rpac_name   = rpac_string.c_str();
    std::string bwt_string  = std::string( output_name ) + ".bwt";
    const char* bwt_name    = bwt_string.c_str();
    std::string rbwt_string = std::string( output_name ) + ".rbwt";
    const char* rbwt_name   = rbwt_string.c_str();

    log_info(stderr, "max length : %lld\n", max_length);
    log_info(stderr, "input      : \"%s\"\n", input_name);
    log_info(stderr, "output     : \"%s\"\n", output_name);

    return build( input_name, output_name, pac_name, rpac_name, bwt_name, rbwt_name, max_length );
}

