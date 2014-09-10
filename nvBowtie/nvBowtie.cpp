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

// nvBowtie.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <nvbio/basic/console.h>
#include <nvbio/basic/exceptions.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/io/fmindex/fmindex.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_mmap.h>
#include <nvBowtie/bowtie2/cuda/bowtie2_cuda_driver.h>

void crcInit();

namespace nvbio {
namespace bowtie2 {
namespace cuda {

    void test_seed_hit_deques();
    void test_scoring_queues();

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

using namespace nvbio;

// bogus implementation of a function to check if a string is a number
bool is_number(const char* str)
{
    if (str[0] == '-')
        ++str;

    for (;*str != '\0'; ++str)
    {
        if (*str < '0' ||
            *str > '9')
            return false;
    }
    return true;
}

int main(int argc, char* argv[])
{
    cudaSetDeviceFlags( cudaDeviceMapHost | cudaDeviceLmemResizeToMax );

    crcInit();

    if (argc == 1 ||
        (argc == 2 && strcmp( argv[1], "--help" ) == 0) ||
        (argc == 2 && strcmp( argv[1], "-h" ) == 0))
    {
        log_info(stderr,"nvBowtie [options] reference-genome read-file output\n");
        log_info(stderr,"options:\n");
        log_info(stderr,"  General:\n");
        log_info(stderr,"    --max-reads        int [-1]      maximum number of reads to process\n");
        log_info(stderr,"    --device           int [0]       select the given cuda device\n");
        log_info(stderr,"    --file-ref                       load reference from file\n");
        log_info(stderr,"    --server-ref                     load reference from server\n");
        log_info(stderr,"    --phred33                        qualities are ASCII characters equal to Phred quality + 33\n");
        log_info(stderr,"    --phred64                        qualities are ASCII characters equal to Phred quality + 64\n");
        log_info(stderr,"    --solexa-quals                   qualities are in the Solexa format\n");
        log_info(stderr,"    --pe                             paired ends input\n");
        log_info(stderr,"    --ff                             paired mates are forward-forward\n");
        log_info(stderr,"    --fr                             paired mates are forward-reverse\n");
        log_info(stderr,"    --rf                             paired mates are reverse-forward\n");
        log_info(stderr,"    --rr                             paired mates are reverse-reverse\n");
        log_info(stderr,"    --verbosity                      verbosity level\n");
        log_info(stderr,"  Seeding:\n");
        log_info(stderr,"    --seed-len         int   [22]    seed lengths\n");
        log_info(stderr,"    --seed-freq        float [1.15]  seed spacing, specified as 1 + X*sqrt(read-len)\n");
        log_info(stderr,"    --max-hits         int   [100]   maximum amount of seed hits\n");
        log_info(stderr,"    --max-reseed       int   [2]     number of reseeding rounds\n");
        log_info(stderr,"  Extension:\n");
        log_info(stderr,"    --rand                           randomized seed selection\n");
        log_info(stderr,"    --max-dist         int [15]      maximum edit distance\n");
        log_info(stderr,"    --max-effort-init  int [15]      initial maximum number of consecutive extension failures\n");
        log_info(stderr,"    --max-effort       int [15]      maximum number of consecutive extension failures\n");
        log_info(stderr,"    --min-ext          int [30]      minimum number of extensions per read\n");
        log_info(stderr,"    --max-ext          int [400]     maximum number of extensions per read\n");
        log_info(stderr,"    --minins           int [0]       minimum insert length\n");
        log_info(stderr,"    --minins           int [500]     maximum insert length\n");
        log_info(stderr,"    --overlap                        allow overlapping mates\n");
        log_info(stderr,"    --dovetail                       allow dovetailing mates\n");
        log_info(stderr,"    --no-mixed                       only report paired alignments\n");
        log_info(stderr,"  Reporting:\n");
        log_info(stderr,"    --mapQ-filter      int [0]       minimum mapQ threshold\n");
        exit(0);
    }
    else if (argc == 2 && strcmp( argv[1], "-test" ) == 0)
    {
        log_visible(stderr, "nvBowtie tests... started\n");
        nvbio::bowtie2::cuda::test_seed_hit_deques();
        nvbio::bowtie2::cuda::test_scoring_queues();
        log_visible(stderr, "nvBowtie tests... done\n");
        exit(0);
    }

    uint32 max_reads    = uint32(-1);
    uint32 max_read_len = uint32(-1);
    //bool   debug        = false;
    int    cuda_device  = -1;
    bool   from_file    = false;
    bool   paired_end   = false;
    io::PairedEndPolicy pe_policy = io::PE_POLICY_FR;
    io::QualityEncoding qencoding = io::Phred33;

    std::map<std::string,std::string> string_options;

    for (int32 i = 1; i < argc; ++i)
    {
        if (strcmp( argv[i], "--pe" ) == 0 ||
            strcmp( argv[i], "-paired-ends" ) == 0 ||
            strcmp( argv[i], "--paired-ends" ) == 0)
            paired_end = true;
        else if (strcmp( argv[i], "--ff" ) == 0)
            pe_policy = io::PE_POLICY_FF;
        else if (strcmp( argv[i], "--fr" ) == 0)
            pe_policy = io::PE_POLICY_FR;
        else if (strcmp( argv[i], "--rf" ) == 0)
            pe_policy = io::PE_POLICY_RF;
        else if (strcmp( argv[i], "--rr" ) == 0)
            pe_policy = io::PE_POLICY_RR;
        else if (strcmp( argv[i], "-max-reads" ) == 0 ||
                 strcmp( argv[i], "--max-reads" ) == 0)
            max_reads = atoi( argv[++i] );
        else if (strcmp( argv[i], "-max-read-len" ) == 0 ||
                 strcmp( argv[i], "--max-read-len" ) == 0)
            max_read_len = atoi( argv[++i] );
        else if (strcmp( argv[i], "-file-ref" ) == 0 ||
                 strcmp( argv[i], "--file-ref" ) == 0)
            from_file = true;
        else if (strcmp( argv[i], "-server-ref" ) == 0 ||
                 strcmp( argv[i], "--server-ref" ) == 0)
            from_file = false;
        else if (strcmp( argv[i], "-input" ) == 0 ||
                 strcmp( argv[i], "--input" ) == 0)
        {
            if (strcmp( argv[i+1], "file" ) == 0)
                from_file = true;
            else if (strcmp( argv[i+1], "server" ) == 0)
                from_file = false;
            else
                log_warning(stderr, "unknown \"%s\" input, skipping\n", argv[i+1]);

            ++i;
        }
        else if (strcmp( argv[i], "-phred33" ) == 0 ||
                 strcmp( argv[i], "--phred33" ) == 0)
            qencoding = io::Phred33;
        else if (strcmp( argv[i], "-phred64" ) == 0 ||
                 strcmp( argv[i], "--phred64" ) == 0)
            qencoding = io::Phred64;
        else if (strcmp( argv[i], "-solexa-quals" ) == 0 ||
                 strcmp( argv[i], "--solexa-quals" ) == 0)
            qencoding = io::Solexa;
        // xxxnsubtil: debug seems to be set but never used
//        else if (strcmp( argv[i], "-debug" ) == 0)
//            debug = true;
        else if (strcmp( argv[i], "-device" ) == 0 ||
                 strcmp( argv[i], "--device" ) == 0)
            cuda_device = atoi( argv[++i] );
        else if (strcmp( argv[i], "-verbosity" ) == 0 ||
                 strcmp( argv[i], "--verbosity" ) == 0)
            set_verbosity( Verbosity( atoi( argv[++i] ) ) );
        else if (argv[i][0] == '-')
        {
            // add unknown option to the string options
            const std::string key = std::string( argv[i][1] == '-' ? argv[i] + 2 : argv[i] + 1 );
            const char* next = argv[i+1];

            if (is_number(next) || next[0] != '-')
            {
                const std::string val = std::string( next ); ++i;
                string_options.insert( std::make_pair( key, val ) );
            }
            else
                string_options.insert( std::make_pair( key, "1" ) );
        }
    }

    log_info(stderr, "nvBowtie... started\n");
    log_debug(stderr, "  %-16s : %d\n", "max-reads",  max_reads);
    log_debug(stderr, "  %-16s : %d\n", "max-length", max_read_len);
    log_debug(stderr, "  %-16s : %s\n", "quals", qencoding == io::Phred33 ? "phred33" :
                                                 qencoding == io::Phred64 ? "phred64" :
                                                                            "solexa");
    if (paired_end)
    {
        log_debug(stderr, "  %-16s : %s\n", "pe-policy",
            pe_policy == io::PE_POLICY_FF ? "ff" :
            pe_policy == io::PE_POLICY_FR ? "fr" :
            pe_policy == io::PE_POLICY_RF ? "rf" :
                                            "rr" );
    }
    if (string_options.empty() == false)
    {
        for (std::map<std::string,std::string>::const_iterator it = string_options.begin(); it != string_options.end(); ++it)
            log_debug(stderr, "  %-16s : %s\n", it->first.c_str(), it->second.c_str());
    }
    log_debug(stderr, "\n");

    int device_count;
    cudaGetDeviceCount(&device_count);
    log_verbose(stderr, "  cuda devices : %d\n", device_count);

    // inspect and select cuda devices
    if (device_count)
    {
        if (cuda_device == -1)
        {
            int            best_device = 0;
            cudaDeviceProp best_device_prop;
            cudaGetDeviceProperties( &best_device_prop, best_device );

            for (int device = 0; device < device_count; ++device)
            {
                cudaDeviceProp device_prop;
                cudaGetDeviceProperties( &device_prop, device );
                log_verbose(stderr, "  device %d has compute capability %d.%d\n", device, device_prop.major, device_prop.minor);
                log_verbose(stderr, "    SM count          : %u\n", device_prop.multiProcessorCount);
                log_verbose(stderr, "    SM clock rate     : %u Mhz\n", device_prop.clockRate / 1000);
                log_verbose(stderr, "    memory clock rate : %.1f Ghz\n", float(device_prop.memoryClockRate) * 1.0e-6f);

                if (device_prop.major >= best_device_prop.major &&
                    device_prop.minor >= best_device_prop.minor)
                {
                    best_device_prop = device_prop;
                    best_device      = device;
                }
            }
            cuda_device = best_device;
        }
        log_verbose(stderr, "  chosen device %d\n", cuda_device);
        {
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties( &device_prop, cuda_device );
            log_verbose(stderr, "    device name        : %s\n", device_prop.name);
            log_verbose(stderr, "    compute capability : %d.%d\n", device_prop.major, device_prop.minor);
        }
        cudaSetDevice( cuda_device );
    }

    uint32 arg_offset = paired_end ? argc-4 : argc-3;

    try
    {
        SharedPointer<nvbio::io::SequenceData> reference_data;
        SharedPointer<nvbio::io::FMIndexData>  driver_data;
        if (from_file)
        {
            log_visible(stderr, "loading reference index... started\n");
            log_info(stderr, "  file: \"%s\"\n", argv[arg_offset]);

            // load the reference data
            reference_data = io::load_sequence_file( DNA, argv[arg_offset] );
            if (reference_data == NULL)
            {
                log_error(stderr, "unable to load reference index \"%s\"\n", argv[arg_offset]);
                return 1;
            }

            log_visible(stderr, "loading reference index... done\n");

            nvbio::io::FMIndexDataHost* loader = new nvbio::io::FMIndexDataHost;
            if (!loader->load( argv[arg_offset] ))
                return 1;

            driver_data = loader;
        }
        else
        {
            log_visible(stderr, "mapping reference index... started\n");
            log_info(stderr, "  file: \"%s\"\n", argv[arg_offset]);

            // map the reference data
            reference_data = io::map_sequence_file( argv[arg_offset] );
            if (reference_data == NULL)
            {
                log_error(stderr, "mapping reference index \"%s\" failed\n", argv[arg_offset]);
                return 1;
            }

            log_visible(stderr, "mapping reference index... done\n");

            nvbio::io::FMIndexDataMMAP* loader = new nvbio::io::FMIndexDataMMAP;
            if (!loader->load( argv[arg_offset] ))
                return 1;

            driver_data = loader;
        }

        if (paired_end)
        {
            log_visible(stderr, "opening read file [1] \"%s\"\n", argv[arg_offset+1]);
            SharedPointer<nvbio::io::SequenceDataStream> read_data_file1(
                nvbio::io::open_sequence_file(argv[arg_offset+1],
                                          qencoding,
                                          max_reads,
                                          max_read_len,
                                          io::REVERSE)
            );

            if (read_data_file1 == NULL || read_data_file1->is_ok() == false)
            {
                log_error(stderr, "unable to open read file \"%s\"\n", argv[arg_offset+1]);
                return 1;
            }

            log_visible(stderr, "opening read file [2] \"%s\"\n", argv[arg_offset+2]);
            SharedPointer<nvbio::io::SequenceDataStream> read_data_file2(
                nvbio::io::open_sequence_file(argv[arg_offset+2],
                                          qencoding,
                                          max_reads,
                                          max_read_len,
                                          io::REVERSE)
            );

            if (read_data_file2 == NULL || read_data_file2->is_ok() == false)
            {
                log_error(stderr, "unable to open read file \"%s\"\n", argv[arg_offset+2]);
                return 1;
            }

            nvbio::bowtie2::cuda::driver( argv[argc-1], *reference_data, *driver_data, pe_policy, *read_data_file1, *read_data_file2, string_options );
        }
        else
        {
            log_visible(stderr, "opening read file \"%s\"\n", argv[arg_offset+1]);
            SharedPointer<nvbio::io::SequenceDataStream> read_data_file(
                nvbio::io::open_sequence_file(argv[arg_offset+1],
                                          qencoding,
                                          max_reads,
                                          max_read_len,
                                          io::REVERSE)
            );

            if (read_data_file == NULL || read_data_file->is_ok() == false)
            {
                log_error(stderr, "unable to open read file \"%s\"\n", argv[arg_offset+1]);
                return 1;
            }

            nvbio::bowtie2::cuda::driver( argv[argc-1], *reference_data, *driver_data, *read_data_file, string_options );
        }

        log_info( stderr, "nvBowtie... done\n" );
    }
    catch (nvbio::cuda_error e)
    {
        log_error(stderr, "caught a nvbio::cuda_error exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (nvbio::bad_alloc e)
    {
        log_error(stderr, "caught a nvbio::bad_alloc exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (nvbio::logic_error e)
    {
        log_error(stderr, "caught a nvbio::logic_error exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (nvbio::runtime_error e)
    {
        log_error(stderr, "caught a nvbio::runtime_error exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (std::bad_alloc e)
    {
        log_error(stderr, "caught a std::bad_alloc exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (std::logic_error e)
    {
        log_error(stderr, "caught a std::logic_error exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (std::runtime_error e)
    {
        log_error(stderr, "caught a std::runtime_error exception:\n");
        log_error(stderr, "  %s\n", e.what());
    }
    catch (...)
    {
        log_error(stderr, "caught an unknown exception!\n");
    }

	return 0;
}

