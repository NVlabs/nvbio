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
#include <nvbio/basic/bnt.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/timer.h>
#include <nvbio/fmindex/dna.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fasta/fasta.h>
#include <libdivsufsortxx/divsufsortxx.h>
#include "fake_vector.h"
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
typedef int32 SA_storage_type;
typedef int32 SA_facade_type;
#elif (SA_REP == _64_64)
typedef int64 SA_storage_type;
typedef int64 SA_facade_type;
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
        const uint32 n_words = (uint32)min( uint64(1024u), seq_words - words );
        if (fwrite( stream + words, sizeof(StreamType), n_words, output_file ) != n_words)
            return false;
    }
    return true;
}

template <typename StreamType>
int perform(
    const char*  input_name,
    const char*  output_name,
    const char*  pac_name,
    const char*  rpac_name,
    const char*  bwt_name,
    const char*  rbwt_name,
    const uint32 lib,
    const uint64 max_length)
{
    std::vector<std::string> sortednames;
    list_files(input_name, sortednames);
/*
    std::string base_input_name = input_name;

    std::string base_path = base_input_name.substr( 0, base_input_name.rfind('\\')+1 );
    fprintf(stderr, "directory  : \"%s\"\n", base_path.c_str());

    uint32 n_inputs = 0;
    std::string inputnames[1024];

    _finddata_t file_info;
    intptr_t find_handle = _findfirst( base_input_name.c_str(), &file_info );
    if (find_handle == -1)
    {
        fprintf(stderr, "unable to locate \"%s\"", base_input_name.c_str());
        exit(1);
    }

    inputnames[ n_inputs++ ] = base_path + std::string( file_info.name );
    while (_findnext( find_handle, &file_info ) != -1)
        inputnames[ n_inputs++ ] = base_path + std::string( file_info.name );

    // sort files...
    std::pair<uint32,uint32> nums[1024];
    for (uint32 i = 0; i < n_inputs; ++i)
    {
        size_t pos2 = inputnames[i].rfind('.');
        size_t pos1 = inputnames[i].rfind('.', pos2-1);

        std::string numstring = inputnames[i].substr( pos1+1, pos2 - pos1 - 1 );

        nums[i].first  = atoi( numstring.c_str() );
        nums[i].second = i;
    }
    std::sort( nums, nums + n_inputs );
*/
    uint32 n_inputs = (uint32)sortednames.size();
    fprintf(stderr, "\ncounting bps... started\n");
    // count entire sequence length
    Counter counter;

    for (uint32 i = 0; i < n_inputs; ++i)
    {
        fprintf(stderr, "  counting \"%s\"\n", sortednames[i].c_str());

        FASTA_inc_reader fasta( sortednames[i].c_str() );
        if (fasta.valid() == false)
        {
            fprintf(stderr, "  error: unable to open file\n");
            exit(1);
        }

        while (fasta.read( 1024, counter ) == 1024);
    }
    fprintf(stderr, "counting bps... done\n");

    const uint64 seq_length   = nvbio::min( (uint64)counter.m_size, (uint64)max_length );
    const uint32 bps_per_word = sizeof(StreamType)*4u;
    const uint32 words_per_32 = sizeof(uint32)/sizeof(StreamType);
    const uint64 seq_words    = (seq_length + bps_per_word - 1u) / bps_per_word;
    const uint64 seq_words32  = (seq_words  + words_per_32 - 1u) / words_per_32;
    const uint64 sa_words     = lib == BWTSW ? 0u : seq_length+1u;

    fprintf(stderr, "\nstats:\n");
    fprintf(stderr, "  reads           : %u\n", counter.m_reads );
    fprintf(stderr, "  sequence length : %llu bps (%.1f MB)\n",
        seq_length,
        float(seq_words32*sizeof(uint32))/float(1024*1024));
    fprintf(stderr, "  buffer size     : %.1f MB\n",
        float(sa_words*sizeof(SA_storage_type) + 2*seq_words32*sizeof(uint32))/1.0e6f );

    // allocate the actual storage
    uint8* buffer = (uint8*)malloc( sa_words*sizeof(SA_storage_type) + (2*seq_words32)*sizeof(uint32) );
    if (buffer == NULL)
    {
        fprintf(stderr, "  error: unable to allocate buffer!\n");
        exit(1);
    }

    SA_storage_type* bwt_temp    = (SA_storage_type*)( &buffer[0] );
    StreamType*      base_stream = (StreamType*)((SA_storage_type*)( &buffer[0] ) + sa_words);
    uint32*          bwt_stream  = ((uint32*)base_stream) + seq_words32;

    typedef PackedStream<StreamType*,uint8,2,true,SA_facade_type> stream_type;
    typedef PackedStream<uint32*,uint8,2,true,SA_facade_type> bwt_stream_type;
    stream_type     stream( base_stream );
    bwt_stream_type bwt( bwt_stream );

    fprintf(stderr, "\nbuffering bps... started\n");
    // read all files
    {
        Writer<StreamType> writer( base_stream, counter.m_reads, seq_length );

        for (uint32 i = 0; i < n_inputs; ++i)
        {
            fprintf(stderr, "  buffering \"%s\"\n", sortednames[i].c_str());

            FASTA_inc_reader fasta( sortednames[i].c_str() );
            if (fasta.valid() == false)
            {
                fprintf(stderr, "  error: unable to open file!\n");
                exit(1);
            }

            while (fasta.read( 1024, writer ) == 1024);
        }

        save_bns( writer.m_bntseq, output_name );
    }
    fprintf(stderr, "buffering bps... done\n");
    {
        const uint32 crc = crcCalc( stream.begin(), uint32(seq_length) );
        fprintf(stderr, "  crc: %u\n", crc);
    }

    // writing
    if (pac_name)
    {
        fprintf(stderr, "\nwriting \"%s\"... started\n", pac_name);
        if (sizeof(StreamType) == 4)
        {
            //
            // .wpac file
            //

            FILE* output_file = fopen( pac_name, "wb" );
            if (output_file == NULL)
            {
                fprintf(stderr, "  error: could not open output file \"%s\"!\n", pac_name );
                exit(1);
            }

            fwrite( &seq_length, sizeof(uint64), 1, output_file );
            if (save_stream( output_file, seq_words, base_stream ) == false)
            {
                free( buffer );
                fprintf(stderr, "error: writing failed!\n");
                exit(1);
            }

            fclose( output_file );
        }
        else
        {
            //
            // .pac file
            //

            FILE* output_file = fopen( pac_name, "wb" );
            if (output_file == NULL)
            {
                fprintf(stderr, "  error: could not open output file \"%s\"!\n", pac_name );
                exit(1);
            }

            if (save_stream( output_file, seq_words, base_stream ) == false)
            {
                free( buffer );
                fprintf(stderr, "error: writing failed!\n");
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
                fprintf(stderr, "  error: could not open output file \"%s\"!\n", rpac_name );
                exit(1);
            }

            StreamType* rbase_stream = (StreamType*)bwt_stream;

            typedef PackedStream<StreamType*,uint8,2,true,SA_facade_type> stream_type;
            stream_type stream( base_stream );
            stream_type rstream( rbase_stream );

            for (uint32 i = 0; i < seq_length; ++i)
                rstream[i] = stream[ seq_length - i - 1u ];

            if (save_stream( output_file, seq_words, rbase_stream ) == false)
            {
                free( buffer );
                fprintf(stderr, "error: writing failed!\n");
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
        }
        fprintf(stderr, "writing \"%s\"... done\n", pac_name);
    }

    uint32 primary;

    Timer timer;

    if (lib == BWTSW)
    {
        fprintf(stderr, "\nbuilding BWT... started\n");
        timer.start();

        bwt_bwtgen( pac_name, bwt_name );

        timer.stop();
        fprintf(stderr, "building BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);

        fprintf(stderr, "\nbuilding reverse BWT... started\n");
        timer.start();

        bwt_bwtgen( rpac_name, rbwt_name );

        timer.stop();
        fprintf(stderr, "building reverse BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);
    }
    else
    {
        fake_vector<SA_facade_type,SA_storage_type> bwt_vec( bwt_temp );

        fprintf(stderr, "\nbuilding BWT... started\n");
        timer.start();

        try
        {
            if (lib == DIVSUFSORT)
            {
                primary = (uint32)divsufsortxx::constructBWT(
                    stream.begin(), stream.begin() + SA_facade_type(seq_length),
                    bwt.begin(), bwt.begin() + SA_facade_type(seq_length),
                #if SA_REP == _32_64
                    bwt_vec.begin(),
                    bwt_vec.begin() + seq_length,
                #else
                    bwt_temp,
                    bwt_temp + seq_length,
                #endif
                    4 );
            }
            else
            {
                primary = (uint32)saisxx_bwt(
                    stream.begin(),
                    bwt.begin(),
                    bwt_vec.begin(),
                    SA_facade_type(seq_length),
                    SA_facade_type(4) );
            }
        }
        catch (fake_vector_out_of_range)
        {
            fprintf(stderr,"error: value out of range!\n");
            exit(1);
        }
        catch (...)
        {
            fprintf(stderr,"error: unknown exception!\n");
            exit(1);
        }

        uint32 cumFreq[4] = { 0, 0, 0, 0 };

        timer.stop();
        fprintf(stderr, "building BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);
        fprintf(stderr, "  primary: %u\n", primary);
        {
            const uint32 crc = crcCalc( bwt.begin(), uint32(seq_length) );
            fprintf(stderr, "  crc: %u\n", crc);
        }

        FILE* output_file = fopen( bwt_name, "wb" );
        if (output_file == NULL)
        {
            fprintf(stderr, "  error: could not open output file \"%s\"!\n", bwt_name );
            exit(1);
        }

        fprintf(stderr, "\nwriting \"%s\"... started\n", bwt_name);
        fwrite( &primary, sizeof(uint32), 1, output_file );
        fwrite( cumFreq,  sizeof(uint32), 4, output_file );
        if (save_stream( output_file, seq_words32, bwt_stream ) == false)
        {
            free( buffer );
            fprintf(stderr, "error: writing failed!\n");
            exit(1);
        }
        fprintf(stderr, "writing \"%s\"... done\n", bwt_name);
        fclose( output_file );

        fprintf(stderr, "\nbuffering bps... started\n");
        // read all files again
        {
            Writer<StreamType> writer( base_stream, counter.m_reads, seq_length );

            for (uint32 i = 0; i < n_inputs; ++i)
            {
                fprintf(stderr, "  buffering \"%s\"\n", sortednames[i].c_str());

                FASTA_inc_reader fasta( sortednames[i].c_str() );
                if (fasta.valid() == false)
                {
                    fprintf(stderr, "  error: unable to open file!\n");
                    exit(1);
                }

                while (fasta.read( 1024, writer ) == 1024);
            }
        }
        fprintf(stderr, "buffering bps... done\n");

        typedef StreamRemapper< stream_type, reverse_functor<SA_facade_type> > rstream_type;

        rstream_type rstream( stream, reverse_functor<SA_facade_type>( SA_facade_type(seq_length) ) );

        fprintf(stderr, "\nbuilding reverse BWT... started\n");
        timer.start();

        try {
            if (lib == DIVSUFSORT)
            {
                primary = (uint32)divsufsortxx::constructBWT(
                    rstream.begin(), rstream.begin() + SA_facade_type(seq_length),
                    bwt.begin(), bwt.begin() + SA_facade_type(seq_length),
                #if (SA_REP == _32_64)
                    bwt_vec.begin(),
                    bwt_vec.begin() + seq_length,
                #else
                    bwt_temp,
                    bwt_temp + seq_length,
                #endif
                    4 );
            }
            else
            {
                primary = (uint32)saisxx_bwt(
                    rstream.begin(),
                    bwt.begin(),
                    bwt_vec.begin(),
                    SA_facade_type(seq_length),
                    SA_facade_type(4) );
            }
        }
        catch (fake_vector_out_of_range)
        {
            fprintf(stderr,"error: value out of range!\n");
            exit(1);
        }
        catch (...)
        {
            fprintf(stderr,"error: unknown exception!\n");
            exit(1);
        }

        timer.stop();
        fprintf(stderr, "building reverse BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);
        fprintf(stderr, "  primary: %u\n", primary);
        {
            const uint32 crc = crcCalc( bwt.begin(), uint32(seq_length) );
            fprintf(stderr, "  crc: %u\n", crc);
        }

        fopen( rbwt_name, "wb" );
        if (output_file == NULL)
        {
            fprintf(stderr, "  error: could not open output file \"%s\"!\n", rbwt_name );
            exit(1);
        }

        fprintf(stderr, "\nwriting \"%s\"... started\n", rbwt_name);
        fwrite( &primary, sizeof(uint32), 1, output_file );
        fwrite( cumFreq,  sizeof(uint32), 4, output_file );
        if (save_stream( output_file, seq_words32, bwt_stream ) == false)
        {
            free( buffer );
            fprintf(stderr, "error: writing failed!\n");
            exit(1);
        }
        fprintf(stderr, "writing \"%s\"... done\n", rbwt_name);
        fclose( output_file );
    }

    free( buffer );
    return 0;
}

int main(int argc, char* argv[])
{
    crcInit();

    if (argc < 2)
    {
        fprintf(stderr, "please specify input and output file names, e.g:\n");
        fprintf(stderr, "  nvBWT [options] myinput.*.fa output-prefix\n");
        fprintf(stderr, "  options:\n");
        fprintf(stderr, "    -m     max_length\n");
        fprintf(stderr, "    -lib   divsufsort|sais|bwtsw\n");
        fprintf(stderr, "    -p     byte|word\n");
    }
    fprintf(stderr, "arch       : %lu bit\n", sizeof(void*)*8u);
    fprintf(stderr, "SA storage : %lu bits\n", sizeof(SA_storage_type)*8u);
    fprintf(stderr, "SA facade  : %lu bits\n", sizeof(SA_facade_type)*8u);

    const char* file_names[2] = { NULL, NULL };
    uint64 max_length = uint64(-1);
    uint32 lib        = BWTSW;
    uint32 packing    = BYTE_PACKING;

    uint32 n_files = 0;
    for (int32 i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp( arg, "-m" ) == 0)
        {
            max_length = atoi( argv[i+1] );
            ++i;
        }
        else if (strcmp( arg, "-lib" ) == 0)
        {
            if (strcmp( argv[i+1], "sais" ) == 0)
                lib = SAIS;
            else if (strcmp( argv[i+1], "bwtsw" ) == 0)
                lib = BWTSW;
            else
                lib = DIVSUFSORT;
            ++i;
        }
        else if (strcmp( arg, "-p" ) == 0)
        {
            if (strcmp( argv[i+1], "word" ) == 0)
                packing = WORD_PACKING;
            else
                packing = BYTE_PACKING;
            ++i;
        }
        else
            file_names[ n_files++ ] = argv[i];
    }

    const char* input_name  = file_names[0];
    const char* output_name = file_names[1];
    std::string pac_string  = std::string( output_name ) + (packing == WORD_PACKING ? ".wpac" : ".pac");
    const char* pac_name    = pac_string.c_str();
    std::string rpac_string = std::string( output_name ) + (packing == WORD_PACKING ? ".rwpac" : ".rpac");
    const char* rpac_name   = rpac_string.c_str();
    std::string bwt_string  = std::string( output_name ) + ".bwt";
    const char* bwt_name    = bwt_string.c_str();
    std::string rbwt_string = std::string( output_name ) + ".rbwt";
    const char* rbwt_name   = rbwt_string.c_str();

    fprintf(stderr, "lib        : %s\n", lib == BWTSW ? "bwtsw" : lib == DIVSUFSORT ? "divsufsort" : "sais");
    fprintf(stderr, "packing    : %s\n", packing == BYTE_PACKING ? "byte" : "word");
    fprintf(stderr, "max length : %lld\n", max_length);
    fprintf(stderr, "input      : \"%s\"\n", input_name);
    fprintf(stderr, "output     : \"%s\"\n", output_name);

    if (packing == BYTE_PACKING)
        return perform<uint8>( input_name, output_name, pac_name, rpac_name, bwt_name, rbwt_name, lib, max_length );
    else if (packing == WORD_PACKING)
        return perform<uint32>( input_name, output_name, pac_name, rpac_name, bwt_name, rbwt_name, lib, max_length );
}

