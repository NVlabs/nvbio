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

#include <nvbio/basic/console.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/basic/exceptions.h>
#include <nvbio/basic/cuda/arch.h>          // cuda::check_error
#include <nvbio/io/fmi.h>
#include <nvbio/io/output/output_file.h>
#include <nvbio/io/reads/reads.h>

#include "options.h"
#include "util.h"
#include "pipeline.h"
#include "mem-search.h"
#include "build-chains.h"

using namespace nvbio;

int run(int argc, char **argv)
{
    parse_command_line(argc, argv);
    gpu_init();

    struct pipeline_context pipeline;
    // load the fmindex and prepare the SMEM search
    mem_init(&pipeline);

    // open the input read file
    SharedPointer<io::ReadDataStream> input = SharedPointer<io::ReadDataStream>(
        io::open_read_file(
            command_line_options.input_file_name,
            io::Phred33,
            uint32(-1),
            uint32(-1),
            io::ReadEncoding(io::FORWARD | io::REVERSE_COMPLEMENT) ) );

    if (input == NULL || input->is_ok() == false)
    {
        log_error(stderr, "failed to open read file %s\n", command_line_options.input_file_name);
        exit(1);
    }

    // open the output file
    pipeline.output = io::OutputFile::open(command_line_options.output_file_name,
            io::SINGLE_END,
            io::BNT(*pipeline.mem.fmindex_data_host));

    if (!pipeline.output)
    {
        log_error(stderr, "failed to open output file %s\n", command_line_options.output_file_name);
        exit(1);
    }

    Timer  global_timer;
    Timer  timer;

    // go!
    for(;;)
    {
        global_timer.start();
        timer.start();

        // read the next batch
        SharedPointer<io::ReadData> batch = SharedPointer<io::ReadData>( input->next(command_line_options.batch_size, uint32(-1)) );
        if (batch == NULL)
        {
            // EOF
            break;
        }
        log_info(stderr, "processing reads [%llu,%llu)\n", pipeline.stats.n_reads, pipeline.stats.n_reads + batch->size()/2);

        // copy batch to the device
        const io::ReadDataDevice device_batch(*batch);

        timer.stop();
        pipeline.stats.io_time += timer.seconds();

        // search for MEMs
        mem_search(&pipeline, &device_batch);

        // now start a loop where we break the read batch into smaller chunks for
        // which we can locate all MEMs and build all chains
        for (uint32 read_begin = 0; read_begin < batch->size(); read_begin = pipeline.chunk.read_end)
        {
            // determine the next chunk of reads to process
            fit_read_chunk(&pipeline, &device_batch, read_begin);

            log_verbose(stderr, "processing chunk\n");
            log_verbose(stderr, "  reads : [%u,%u)\n", pipeline.chunk.read_begin, pipeline.chunk.read_end);
            log_verbose(stderr, "  mems  : [%u,%u)\n", pipeline.chunk.mem_begin,  pipeline.chunk.mem_end);

            // locate all MEMs in the current chunk
            mem_locate(&pipeline, &device_batch);

            // build the chains
            build_chains(&pipeline, &device_batch);
        }
        global_timer.stop();
        pipeline.stats.n_reads += batch->size()/2;
        pipeline.stats.time    += global_timer.seconds();

        log_stats(stderr, "  time: %5.1fs (%.1f K reads/s)\n", pipeline.stats.time, 1.0e-3f * float(pipeline.stats.n_reads)/pipeline.stats.time);
        log_stats(stderr, "    io     : %6.2f %%\n", 100.0f * pipeline.stats.io_time/pipeline.stats.time);
        log_stats(stderr, "    search : %6.2f %%\n", 100.0f * pipeline.stats.search_time/pipeline.stats.time);
        log_stats(stderr, "    locate : %6.2f %%\n", 100.0f * pipeline.stats.locate_time/pipeline.stats.time);
        log_stats(stderr, "    chain  : %6.2f %%\n", 100.0f * pipeline.stats.chain_time/pipeline.stats.time);
    }

    pipeline.output->close();
    return 0;
}

int main(int argc, char **argv)
{
    try
    {
        return run( argc, argv );
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
