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

// fmmap.cu
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
#include <nvbio/fmindex/filter.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/fmi.h>

#include "alignment.h"
#include "util.h"

using namespace nvbio;

// alignment params
//
struct Params
{
    uint32 seed_len;
    uint32 seed_intv;
    uint32 merge_intv;
};

// query stats
//
struct Stats
{
    Stats() :
        time(0),
        extract_time(0),
        rank_time(0),
        locate_time(0),
        align_time(0),
        reads(0),
        aligned(0),
        queries(0),
        occurrences(0) {}

    float   time;
    float   extract_time;
    float   rank_time;
    float   locate_time;
    float   align_time;
    uint64  reads;
    uint64  aligned;
    uint64  queries;
    uint64  occurrences;
};

// transform a (index-pos,seed-id) hit into a diagonal (text-pos = index-pos - seed-pos, read-id)
struct hit_to_diagonal
{
    typedef uint2  argument_type;
    typedef uint2  result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    hit_to_diagonal(const string_set_infix_coord_type* _seed_coords) : seed_coords(_seed_coords) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 operator() (const uint2 hit) const
    {
        const uint32 index_pos = hit.x;
        const uint32 seed_id   = hit.y;

        const string_set_infix_coord_type seed = seed_coords[ seed_id ];

        const uint32 read_pos = infix_begin( make_infix( "", seed ) );
        const uint32 read_id  =   string_id( make_infix( "", seed ) );

        return make_uint2( index_pos - read_pos, read_id );
    }

    const string_set_infix_coord_type* seed_coords;
};

// extract a set of uniformly spaced seeds from a string-set and return it as an InfixSet
//
template <typename system_tag, typename string_set_type>
InfixSet<string_set_type, const string_set_infix_coord_type*>
extract_seeds(
    const string_set_type                                   string_set,         // the input string-set
    const uint32                                            seed_len,           // the seeds length
    const uint32                                            seed_interval,      // the spacing between seeds
    nvbio::vector<system_tag,string_set_infix_coord_type>&  seed_coords)        // the output vector of seed coordinates
{
    // enumerate all seeds
    const uint32 n_seeds = enumerate_string_set_seeds(
        string_set,
        uniform_seeds_functor<>( seed_len, seed_interval ),
        seed_coords );

    // and build the output infix-set
    return InfixSet<string_set_type, const string_set_infix_coord_type*>(
        n_seeds,
        string_set,
        nvbio::plain_view( seed_coords ) );
}

// perform q-gram index mapping
//
template <typename fm_index_type, typename fm_filter_type, typename genome_string>
void map(
    const Params                        params,
          fm_index_type&                fm_index,
          fm_filter_type&               fm_filter,
    const io::ReadDataDevice&           reads,
    const uint32                        genome_len,
    const genome_string                 genome,
    nvbio::vector<device_tag,int16>&    best_scores,
          Stats&                        stats)
{
    typedef io::ReadDataDevice::const_read_string_set_type                      read_string_set_type;
    typedef string_set_infix_coord_type                                         infix_coord_type;
    typedef nvbio::vector<device_tag,infix_coord_type>                          infix_vector_type;
    typedef InfixSet<read_string_set_type, const string_set_infix_coord_type*>  seed_string_set_type;

    // prepare some vectors to store the query qgrams
    infix_vector_type seed_coords;

    Timer timer;
    timer.start();

    const read_string_set_type read_string_set = reads.const_read_string_set();
    const seed_string_set_type seed_string_set = extract_seeds(
        read_string_set,
        params.seed_len,
        params.seed_intv,
        seed_coords );

    cudaDeviceSynchronize();
    timer.stop();
    const float extract_time = timer.seconds();

    stats.queries       += seed_string_set.size();
    stats.extract_time  += extract_time;

    //
    // search the sorted query q-grams with a q-gram filter
    //

    const uint32 batch_size = 32*1024*1024;

    typedef uint2  hit_type;

    // prepare storage for the output hits
    nvbio::vector<device_tag,hit_type>      hits( batch_size );
    nvbio::vector<device_tag,int16>         scores( batch_size );
    nvbio::vector<device_tag,uint32>        out_reads( batch_size );
    nvbio::vector<device_tag,int16>         out_scores( batch_size );
    nvbio::vector<device_tag,uint8>         temp_storage;

    timer.start();

    // first step: rank the query seeds
    const uint64 n_hits = fm_filter.rank( fm_index, seed_string_set );

    cudaDeviceSynchronize();
    timer.stop();
    stats.rank_time   += timer.seconds();
    stats.occurrences += n_hits;

    // loop through large batches of hits and locate & merge them
    for (uint64 hits_begin = 0; hits_begin < n_hits; hits_begin += batch_size)
    {
        const uint64 hits_end = nvbio::min( hits_begin + batch_size, n_hits );

        timer.start();

        fm_filter.locate(
            hits_begin,
            hits_end,
            hits.begin() );

        cudaDeviceSynchronize();
        timer.stop();
        stats.locate_time += timer.seconds();

        // transform the (index-pos,seed-id) hit coordinates into diagonals, (text-pos = index-pos - seed-pos, read-id)
        thrust::transform(
            hits.begin(),
            hits.begin() + hits_end - hits_begin,
            hits.begin(),
            hit_to_diagonal( nvbio::plain_view( seed_coords ) ) );

        timer.start();

        //const aln::SimpleGotohScheme gotoh( 2, -2, -5, -3 );
        typedef aln::MyersTag<5u> myers_dna5_tag;
        align(
            aln::make_edit_distance_aligner<aln::SEMI_GLOBAL, myers_dna5_tag>(),
            //aln::make_gotoh_aligner<aln::LOCAL>( gotoh ),
            hits_end - hits_begin,
            nvbio::plain_view( hits ),
            nvbio::plain_view( reads ),
            genome_len,
            genome,
            nvbio::plain_view( scores ) );

        cudaDeviceSynchronize();
        timer.stop();
        stats.align_time += timer.seconds();

        // compute the best score for each read in this batch;
        // note that we divide the string-id by 2 to merge results coming from the forward
        // and reverse-complemented strands
        const uint32 n_distinct = cuda::reduce_by_key(
            hits_end - hits_begin,
            thrust::make_transform_iterator(
                nvbio::plain_view( hits ),
                make_composition_functor( divide_by_two(), component_functor<hit_type>( 1u ) ) ), // take the second component divided by 2
            nvbio::plain_view( scores ),
            nvbio::plain_view( out_reads ),
            nvbio::plain_view( out_scores ),
            thrust::maximum<int16>(),
            temp_storage );

        // and keep track of the global best
        update_scores(
            hits_end - hits_begin,
            nvbio::plain_view( out_reads ),
            nvbio::plain_view( out_scores ),
            nvbio::plain_view( best_scores ) );

        log_verbose(stderr, "\r  processed %6.2f %% reads", 100.0f * float( hits_end ) / float( n_hits ));
    }
    log_verbose_cont(stderr, "\n");
}

// main test entry point
//
int main(int argc, char* argv[])
{
    //
    // perform some basic option parsing
    //

    const uint32 batch_bps     = 128*1024*1024;

    const char* reads = argv[argc-1];
    const char* index = argv[argc-2];

    Params params;
    params.seed_len         = 22;
    params.seed_intv        = 10;
    params.merge_intv       = 16;
    uint32 max_reads        = uint32(-1);
    int16  score_threshold  = -20;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-s" ) == 0)
        {
            params.seed_len  = uint32( atoi( argv[++i] ) );
            params.seed_intv = uint32( atoi( argv[++i] ) );
        }
        if (strcmp( argv[i], "-m" ) == 0)
            params.merge_intv = uint32( atoi( argv[++i] ) );
        else if (strcmp( argv[i], "-max-reads" ) == 0)
            max_reads = uint32( atoi( argv[++i] ) );
        else if (strcmp( argv[i], "-t" ) == 0)
            score_threshold = int16( atoi( argv[++i] ) );
    }

    // TODO: load a genome archive...
    io::FMIndexDataRAM h_fmi;
    if (!h_fmi.load( index, io::FMIndexData::GENOME | io::FMIndexData::FORWARD | io::FMIndexData::SA ))
    {
        log_error(stderr, "    failed loading index \"%s\"\n", index);
        return 1u;
    }

    // build its device version
    const io::FMIndexDataDevice d_fmi( h_fmi, io::FMIndexDataDevice::GENOME | io::FMIndexData::FORWARD | io::FMIndexData::SA );

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
            max_reads,
            uint32(-1),
            io::ReadEncoding( io::FORWARD | io::REVERSE_COMPLEMENT ) ) );

    // check whether the file opened correctly
    if (read_data_file == NULL || read_data_file->is_ok() == false)
    {
        log_error(stderr, "    failed opening file \"%s\"\n", reads);
        return 1u;
    }
    log_info(stderr, "  opening reads file... done\n");

    const uint32 batch_size = 1024*1024;

    typedef io::FMIndexDataDevice::fm_index_type        fm_index_type;
    typedef FMIndexFilterDevice<fm_index_type>          fm_filter_type;

    // fetch the FM-index
    const fm_index_type fm_index = d_fmi.index();

    // create an FM-index filter
    fm_filter_type fm_filter;

    // keep stats
    Stats stats;

    while (1)
    {
        // load a batch of reads
        SharedPointer<io::ReadData> h_read_data( read_data_file->next( batch_size, batch_bps ) );
        if (h_read_data == NULL)
            break;

        log_info(stderr, "  loading reads... started\n");

        // copy it to the device
        const io::ReadDataDevice d_read_data( *h_read_data );

        const uint32 n_reads = d_read_data.size() / 2;

        log_info(stderr, "  loading reads... done\n");
        log_info(stderr, "    %u reads\n", n_reads);

        const int16 worst_score = Field_traits<int16>::min();
        nvbio::vector<device_tag,int16> best_scores( n_reads, worst_score );
        nvbio::vector<device_tag,uint8> temp_storage;

        Timer timer;
        timer.start();

        map(
            params,
            fm_index,
            fm_filter,
            d_read_data,
            genome_len,
            d_genome,
            best_scores,
            stats );

        timer.stop();
        const float time = timer.seconds();

        // accumulate the number of aligned reads
        stats.reads += n_reads;
        stats.time  += time;

        // count how many reads have a score >= score_threshold
        const uint32 n_aligned = cuda::reduce(
            n_reads,
            thrust::make_transform_iterator( nvbio::plain_view( best_scores ), above_threshold( score_threshold ) ),
            thrust::plus<uint32>(),
            temp_storage );

        stats.aligned += n_aligned;

        log_info(stderr, "  aligned %6.2f %% reads (%6.2f K reads/s)\n", 100.0f * float( stats.aligned ) / float( stats.reads ), (1.0e-3f * float( stats.reads )) / stats.time);
        log_verbose(stderr, "  breakdown:\n");
        log_verbose(stderr, "    extract throughput : %.2f B seeds/s\n",  (1.0e-9f * float( stats.queries )) / stats.extract_time);
        log_verbose(stderr, "    rank throughput    : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.rank_time);
        log_verbose(stderr, "                       : %6.2f B seeds/s\n", (1.0e-9f * float( stats.queries )) / stats.rank_time);
        log_verbose(stderr, "    locate throughput  : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.locate_time);
        log_verbose(stderr, "    align throughput   : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.align_time);
        log_verbose(stderr, "                       : %6.2f M hits/s\n",  (1.0e-6f * float( stats.occurrences )) / stats.align_time);
        log_verbose(stderr, "    occurrences        : %.3f B\n", 1.0e-9f * float( stats.occurrences ) );
    }
    return 0;
}
