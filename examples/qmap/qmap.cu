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

// qmap.cu
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
#include <nvbio/qgram/qgram.h>
#include <nvbio/qgram/filter.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/fmi.h>

#include "alignment.h"
#include "util.h"

using namespace nvbio;

// query stats
//
struct Stats
{
    Stats() :
        time(0),
        build_time(0),
        extract_time(0),
        rank_time(0),
        locate_time(0),
        align_time(0),
        reads(0),
        aligned(0),
        queries(0),
        matches(0),
        occurrences(0),
        merged(0) {}

    float   time;
    float   build_time;
    float   extract_time;
    float   rank_time;
    float   locate_time;
    float   align_time;
    uint64  reads;
    uint64  aligned;
    uint64  queries;
    uint64  matches;
    uint64  occurrences;
    uint64  merged;
};

// build a set of q-grams from a given string, together with their sorted counterpart
//
template <typename genome_string, typename qgram_vector_type, typename index_vector_type>
void build_qgrams(
    const uint32                    Q,
    const uint32                    genome_len,
    const uint32                    genome_offset,
    const genome_string             genome,
    const uint32                    n_queries,
    qgram_vector_type&              qgrams,
    qgram_vector_type&              sorted_qgrams,
    index_vector_type&              sorted_indices)
{
    // build the q-grams
    qgrams.resize( n_queries );
    generate_qgrams( Q, 2u, genome_len, genome, n_queries, thrust::make_counting_iterator<uint32>(genome_offset), qgrams.begin() );

    // sort the q-grams
    sorted_qgrams = qgrams;
    sorted_indices.resize( n_queries );
    thrust::copy(
        thrust::make_counting_iterator<uint32>(genome_offset),
        thrust::make_counting_iterator<uint32>(genome_offset) + n_queries,
        sorted_indices.begin() );

    thrust::sort_by_key( sorted_qgrams.begin(), sorted_qgrams.end(), sorted_indices.begin() );
}

// build a q-gram set-index from a string-set
//
template <typename string_set_type>
void qgram_set_index_build(
    const uint32            Q,
    const uint32            seed_interval,
    const string_set_type   string_set,
    QGramSetIndexDevice&    qgram_index)
{
    log_verbose(stderr, "  building q-gram set-index... started\n");

    Timer timer;
    timer.start();

    // build the q-gram set index
    qgram_index.build(
        Q,              // q-gram size
        2u,             // implicitly convert N to A
        string_set,
        uniform_seeds_functor<>( Q, seed_interval ),
        12u );

    cudaDeviceSynchronize();
    timer.stop();
    const float time = timer.seconds();

    log_verbose(stderr, "  building q-gram set-index... done\n");
    log_verbose(stderr, "    indexed q-grams : %6.2f M q-grams\n", 1.0e-6f * float( qgram_index.n_qgrams ));
    log_verbose(stderr, "    unique q-grams  : %6.2f M q-grams\n", 1.0e-6f * float( qgram_index.n_unique_qgrams ));
    log_verbose(stderr, "    throughput      : %5.1f M q-grams/s\n", 1.0e-6f * float( qgram_index.n_qgrams ) / time);
    log_verbose(stderr, "    memory usage    : %5.1f MB\n", float( qgram_index.used_device_memory() ) / float(1024*1024) );
}

// perform q-gram index mapping
//
template <typename qgram_index_type, typename qgram_filter_type, typename genome_string>
void map(
          qgram_index_type&             qgram_index,
          qgram_filter_type&            qgram_filter,
    const uint32                        merge_intv,
    const io::ReadDataDevice&           reads,
    const uint32                        n_queries,
    const uint32                        genome_len,
    const uint32                        genome_offset,
    const genome_string                 genome,
    nvbio::vector<device_tag,int16>&    best_scores,
          Stats&                        stats)
{
    typedef typename qgram_index_type::system_tag system_tag;

    // prepare some vectors to store the query qgrams
    nvbio::vector<system_tag,uint64>  qgrams( n_queries );
    nvbio::vector<system_tag,uint64>  sorted_qgrams( n_queries );
    nvbio::vector<system_tag,uint32>  sorted_indices( n_queries );

    const uint32 Q = qgram_index.Q;

    Timer timer;
    timer.start();

    build_qgrams(
        Q,
        genome_len,
        genome_offset,
        genome,
        n_queries,
        qgrams,
        sorted_qgrams,
        sorted_indices );

    cudaDeviceSynchronize();
    timer.stop();
    const float extract_time = timer.seconds();

    stats.queries       += n_queries;
    stats.extract_time  += extract_time;

    //
    // search the sorted query q-grams with a q-gram filter
    //

    const uint32 batch_size = 32*1024*1024;

    typedef typename qgram_filter_type::hit_type        hit_type;
    typedef typename qgram_filter_type::diagonal_type   diagonal_type;

    // prepare storage for the output hits
    nvbio::vector<system_tag,hit_type>      hits( batch_size );
    nvbio::vector<system_tag,diagonal_type> merged_hits( batch_size );
    nvbio::vector<system_tag,uint16>        merged_counts( batch_size );
    nvbio::vector<system_tag,int16>         scores( batch_size );
    nvbio::vector<system_tag,uint32>        out_reads( batch_size );
    nvbio::vector<system_tag,int16>         out_scores( batch_size );
    nvbio::vector<system_tag,uint8>         temp_storage;

    timer.start();

    // first step: rank the query q-grams
    const uint64 n_hits = qgram_filter.rank(
        qgram_index,
        n_queries,
        nvbio::plain_view( sorted_qgrams ),
        nvbio::plain_view( sorted_indices ) );

    cudaDeviceSynchronize();
    timer.stop();
    stats.rank_time   += timer.seconds();
    stats.occurrences += n_hits;

    // loop through large batches of hits and locate & merge them
    for (uint64 hits_begin = 0; hits_begin < n_hits; hits_begin += batch_size)
    {
        const uint64 hits_end = nvbio::min( hits_begin + batch_size, n_hits );

        timer.start();

        qgram_filter.locate(
            hits_begin,
            hits_end,
            hits.begin() );

        const uint32 n_merged = qgram_filter.merge(
            merge_intv,
            hits_end - hits_begin,
            hits.begin(),
            merged_hits.begin(),
            merged_counts.begin() );

        cudaDeviceSynchronize();
        timer.stop();
        stats.locate_time += timer.seconds();
        stats.merged      += n_merged;

        timer.start();

        //const aln::SimpleGotohScheme gotoh( 2, -2, -5, -3 );
        typedef aln::MyersTag<5u> myers_dna5_tag;
        align(
            aln::make_edit_distance_aligner<aln::SEMI_GLOBAL, myers_dna5_tag>(),
            //aln::make_gotoh_aligner<aln::LOCAL>( gotoh ),
            n_merged,
            nvbio::plain_view( merged_hits ),
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
            n_merged,
            thrust::make_transform_iterator(
                nvbio::plain_view( merged_hits ),
                make_composition_functor( divide_by_two(), component_functor<diagonal_type>( 1u ) ) ), // take the second component divided by 2
            nvbio::plain_view( scores ),
            nvbio::plain_view( out_reads ),
            nvbio::plain_view( out_scores ),
            thrust::maximum<int16>(),
            temp_storage );

        // and keep track of the global best
        update_scores(
            n_merged,
            nvbio::plain_view( out_reads ),
            nvbio::plain_view( out_scores ),
            nvbio::plain_view( best_scores ) );
    }
}

// main test entry point
//
int main(int argc, char* argv[])
{
    //
    // perform some basic option parsing
    //

    const uint32 batch_reads   =   1*1024*1024;
    const uint32 batch_bps     = 100*1024*1024;
    const uint32 queries_batch =  16*1024*1024;

    const char* reads = argv[argc-1];
    const char* index = argv[argc-2];

    uint32 Q                = 20;
    uint32 Q_intv           = 10;
    uint32 merge_intv       = 16;
    uint32 max_reads        = uint32(-1);
    int16  score_threshold  = -20;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-q" ) == 0)
        {
            Q      = uint32( atoi( argv[++i] ) );
            Q_intv = uint32( atoi( argv[++i] ) );
        }
        if (strcmp( argv[i], "-m" ) == 0)
            merge_intv = uint32( atoi( argv[++i] ) );
        else if (strcmp( argv[i], "-max-reads" ) == 0)
            max_reads = uint32( atoi( argv[++i] ) );
        else if (strcmp( argv[i], "-t" ) == 0)
            score_threshold = int16( atoi( argv[++i] ) );
    }

    // TODO: load a genome archive...
    io::FMIndexDataRAM h_fmi;
    if (!h_fmi.load( index, io::FMIndexData::GENOME ))
    {
        log_error(stderr, "    failed loading index \"%s\"\n", index);
        return 1u;
    }

    // build its device version
    const io::FMIndexDataDevice d_fmi( h_fmi, io::FMIndexDataDevice::GENOME );

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

    // keep stats
    Stats stats;

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

        // prepare some typedefs for the involved string-sets and infixes
        typedef io::ReadData::const_read_string_set_type                        string_set_type;    // the read string-set
        typedef string_set_infix_coord_type                                     infix_coord_type;   // the infix coordinate type, for string-sets
        typedef nvbio::vector<device_tag,infix_coord_type>                      infix_vector_type;  // the device vector type for infix coordinates
        typedef InfixSet<string_set_type, const string_set_infix_coord_type*>   seed_set_type;      // the infix-set type for representing seeds

        // fetch the actual read string-set
        const string_set_type d_read_string_set = d_read_data.read_string_set();

        // build the q-gram index
        QGramSetIndexDevice qgram_index;

        qgram_set_index_build(
            Q,
            Q_intv,
            d_read_string_set,
            qgram_index );

        typedef QGramFilterDevice<QGramSetIndexDevice,const uint64*,const uint32*> qgram_filter_type;
        qgram_filter_type qgram_filter;

        float time = 0.0f;

        const int16 worst_score = Field_traits<int16>::min();
        nvbio::vector<device_tag,int16> best_scores( n_reads, worst_score );
        nvbio::vector<device_tag,uint8> temp_storage;

        // stream through the genome
        for (uint32 genome_begin = 0; genome_begin < genome_len; genome_begin += queries_batch)
        {
            const uint32 genome_end = nvbio::min( genome_begin + queries_batch, genome_len );

            Timer timer;
            timer.start();

            map(
                qgram_index,
                qgram_filter,
                merge_intv,
                d_read_data,
                genome_end - genome_begin,
                genome_len,
                genome_begin,
                d_genome,
                best_scores,
                stats );

            cudaDeviceSynchronize();
            timer.stop();
            time += timer.seconds();

            const float genome_ratio = float( genome_end ) / float( genome_len );

            log_verbose(stderr, "\r  aligned %5.2f%% of genome (%6.2f K reads/s)", 100.0f * genome_ratio, (1.0e-3f * n_reads) * genome_ratio / time  );
        }
        log_verbose_cont(stderr, "\n");

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
        log_verbose(stderr, "    extract throughput : %.2f B q-grams/s\n", (1.0e-9f * float( stats.queries )) / stats.extract_time);
        log_verbose(stderr, "    rank throughput    : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.rank_time);
        log_verbose(stderr, "                       : %6.2f B seeds/s\n", (1.0e-9f * float( stats.queries )) / stats.rank_time);
        log_verbose(stderr, "    locate throughput  : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.locate_time);
        log_verbose(stderr, "    align throughput   : %6.2f K reads/s\n", (1.0e-3f * float( stats.reads )) / stats.align_time);
        log_verbose(stderr, "                       : %6.2f M hits/s\n",  (1.0e-6f * float( stats.merged )) / stats.align_time);
        log_verbose(stderr, "    occurrences        : %.3f B\n", 1.0e-9f * float( stats.occurrences ) );
        log_verbose(stderr, "    merged occurrences : %.3f B (%.1f %%)\n", 1.0e-9f * float( stats.merged ), 100.0f * float(stats.merged)/float(stats.occurrences));
    }
    return 0;
}
