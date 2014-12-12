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

//#define NVBIO_ENABLE_PROFILING

#define MOD_NAMESPACE
#define MOD_NAMESPACE_BEGIN namespace bowtie2 { namespace driver {
#define MOD_NAMESPACE_END   }}
#define MOD_NAMESPACE_NAME bowtie2::driver

#include <nvBowtie/bowtie2/cuda/bowtie2_cuda_driver.h>
#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/fmindex_def.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/stats.h>
#include <nvBowtie/bowtie2/cuda/persist.h>
#include <nvBowtie/bowtie2/cuda/scoring.h>
#include <nvBowtie/bowtie2/cuda/mapq.h>
#include <nvBowtie/bowtie2/cuda/input_thread.h>
#include <nvBowtie/bowtie2/cuda/aligner.h>
#include <nvBowtie/bowtie2/cuda/aligner_inst.h>
#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/options.h>
#include <nvbio/basic/threads.h>
#include <nvbio/basic/atomics.h>
#include <nvbio/basic/html.h>
#include <nvbio/basic/dna.h>
#include <nvbio/basic/version.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fmindex/ssa.h>
#include <nvbio/fmindex/fmindex.h>
#include <nvbio/fmindex/fmindex_device.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

std::map<std::string,std::string> load_options(const char* name)
{
    std::map<std::string,std::string> options;

    FILE* file = fopen( name, "r" );
    if (file == NULL)
    {
        log_warning( stderr, "failed opening \"%s\"\n", name );
        return options;
    }

    char key[1024];
    char value[1024];

    while (fscanf( file, "%s %s", key, value ) == 2)
        options[ key ] = std::string( value );

    fclose( file );

    return options;
}

// bogus implementation of a function to check if a string is a number
bool is_number(const char* str, uint32 len = uint32(-1))
{
    if (str[0] == '-')
        ++str;

    for (uint32 l = 0; *str != '\0' && l < len; ++l)
    {
        const char c = *str; ++str;
        if (c == '.')             continue;
        if (c >= '0' && c <= '9') continue;
        return false;
    }
    return true;
}

// bogus implementation of a function to check if an option is a function
SimpleFunc parse_function(const char* str, const SimpleFunc def)
{
    if (str[1] != ',')
        return def;

    if (!(str[0] == 'C' ||
          str[0] == 'L' ||
          str[0] == 'G' ||
          str[0] == 'S'))
          return def;

    SimpleFunc ret;
    ret.type = (str[0] == 'C') ? SimpleFunc::LinearFunc :
               (str[0] == 'L') ? SimpleFunc::LinearFunc :
               (str[0] == 'G') ? SimpleFunc::LogFunc    :
                                 SimpleFunc::SqrtFunc;

    std::string nums = std::string( str + 2 );
    const size_t c = nums.find(',');
    if (c == std::string::npos)
        return def;

    if (is_number( nums.c_str(), (uint32)c )      == false) return def;
    if (is_number( nums.c_str() + c + 1 ) == false) return def;

    const std::string num1 = nums.substr( 0, c );
    const std::string num2 = std::string( nums.c_str() + c + 1 );

    ret.k = (float)atof( num1.c_str() );
    ret.m = (float)atof( nums.c_str() + c + 1 );

    // take care of transforming constant functions in linear ones
    if (str[0] == 'C')
    {
        //ret.k += ret.m;
        ret.m = 0.0f;
    }
    return ret;
}

template <typename options_type>
SimpleFunc func_option(const options_type& options, const char* name, const SimpleFunc func)
{
    return (options.find( std::string(name) ) != options.end()) ?
        parse_function( options.find(std::string(name))->second.c_str(), func ) :
        func;
}

template <typename options_type>
SimpleFunc func_option(const options_type& options, const char* name1, const char* name2, const SimpleFunc func)
{
    return
        (options.find( std::string(name1) ) != options.end()) ?
            parse_function( options.find(std::string(name1))->second.c_str(), func ) :
        (options.find( std::string(name2) ) != options.end()) ?
            parse_function( options.find(std::string(name2))->second.c_str(), func ) :
            func;
}

void parse_options(Params& params, const std::map<std::string,std::string>& options, bool init)
{
    const bool   old_local        = params.alignment_type == LocalAlignment;
    const uint32 old_scoring_mode = params.scoring_mode;

    params.mode             = mapping_mode( string_option(options, "mode",    init ? "best" : mapping_mode( params.mode )).c_str() ); // mapping mode
    params.scoring_mode     = scoring_mode( string_option(options, "scoring", init ? "sw"   : scoring_mode( params.scoring_mode )).c_str() ); // scoring mode
    params.alignment_type   = uint_option(options, "local",                 init ? 0u      : params.alignment_type == LocalAlignment ) ? LocalAlignment : EndToEndAlignment;           // local alignment
    params.keep_stats       = (bool)uint_option(options, "stats",           init ? 1u      : params.keep_stats);           // keep stats
    params.max_hits         = uint_option(options, "max-hits",              init ? 100u    : params.max_hits);             // too big = memory exhaustion 
    params.max_dist         = uint_option(options, "max-dist",              init ? 15u     : params.max_dist);             // must be <= MAX_BAND_LEN/2
    params.max_effort_init  = uint_option(options, "max-effort-init",       init ? 15u     : params.max_effort_init);      // initial scoring effort limit
    params.max_effort       = uint_option(options, "max-effort",    "D",    init ? 15u     : params.max_effort);           // scoring effort limit
    params.min_ext          = uint_option(options, "min-ext",               init ? 30u     : params.min_ext);              // min # of extensions
    params.max_ext          = uint_option(options, "max-ext",               init ? 400u    : params.max_ext);              // max # of extensions
    params.max_reseed       = uint_option(options, "max-reseed",    "R",    init ? 2u      : params.max_reseed);           // max # of reseeding rounds
    params.rep_seeds        = uint_option(options, "rep-seeds",             init ? 1000u   : params.rep_seeds);            // reseeding threshold
    params.allow_sub        = uint_option(options, "N",                     init ? 0u      : params.allow_sub);            // allow substitution in seed
    params.mapq_filter      = uint_option(options, "mapQ-filter",   "Q",    init ? 0u      : params.mapq_filter);          // filter anything below this
    params.report           = string_option(options, "report",              init ? ""      : params.report.c_str());       // generate a report file
    params.scoring_file     = string_option(options, "scoring-scheme",      init ? ""      : params.scoring_file.c_str());
    params.randomized       =(bool) uint_option(options, "rand",                  init ? 1u      : params.randomized);           // use randomized selection
    params.randomized       =(bool)!uint_option(options, "no-rand",                               !params.randomized);           // don't use randomized selection
    params.top_seed         = uint_option(options, "top",                   init ? 0u      : params.top_seed);             // explore top seed entirely
    params.min_read_len     = uint_option(options, "min-read-len",          init ? 12u     : params.min_read_len);         // minimum read length
    params.ungapped_mates   = uint_option(options, "ungapped-mates", "ug",  init ? 0u      : params.ungapped_mates);       // ungapped mate alignment

    // force the all-mapping mode with the '--all|-a' option
    if (uint_option(options, "all", "a", params.mode == AllMapping))
        params.mode = AllMapping;

    // force Edit-Distance scoring with the '--ed' option
    if (uint_option(options, "ed", params.scoring_mode == EditDistanceMode))
        params.scoring_mode = EditDistanceMode;

    // force Smith-Waterman scoring with the '--sw' option
    if (uint_option(options, "sw", params.scoring_mode == SmithWatermanMode))
        params.scoring_mode = SmithWatermanMode;

    const bool local = params.alignment_type == LocalAlignment;

    // set the default seeding values, or reset them if the alignment type has been changed
    if (init || (local != old_local))
    {
        params.seed_len  = local ? 20 : 22u;
        params.seed_freq = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, (local ? 0.75f : 1.15) );
    }

    params.seed_len         = uint_option(options,  "seed-len",      "L",                    params.seed_len);      // no greater than 32
    params.seed_freq        = func_option( options, "seed-freq",     "i",                    params.seed_freq );    // seed interval
    params.subseed_len      = uint_option(options,  "subseed-len",          init ? 0u      : params.subseed_len);   // no greater than 32

    params.pe_overlap    = (bool) uint_option(options, "overlap",          init ? 1u      : params.pe_overlap);            // paired-end overlap
    params.pe_overlap    = (bool)!uint_option(options, "no-overlap",                       !params.pe_overlap);            // paired-end overlap
    params.pe_dovetail   = (bool) uint_option(options, "dovetail",         init ? 0u      : params.pe_dovetail);           // paired-end dovetail
    params.pe_unpaired   = (bool)!uint_option(options, "no-mixed",         init ? 0u      :!params.pe_unpaired);           // paired-end no-mixed
    params.min_frag_len  = uint_option(options, "minins", "I",      init ? 0u      : params.min_frag_len);          // paired-end minimum fragment length
    params.max_frag_len  = uint_option(options, "maxins", "X",      init ? 500u    : params.max_frag_len);          // paired-end maximum fragment length

    // the maximum batch of reads processed in parallel
    params.max_batch_size = uint_option(options, "batch-size",  init ? 1024u : params.max_batch_size );   // maximum batch size

    // internal controls
    params.scoring_window   =       uint_option(options, "scoring-window",   init ? 32u        : params.scoring_window);       // scoring window size
    params.debug.read_id    = (uint32)int_option(options, "debug-read",      init ? -1         : (int32)params.debug.read_id); // debug read id
    params.debug.select     = (bool)uint_option(options, "debug-select",     init ? 0u         : params.debug.select);       // debug select kernel
    params.debug.locate     = (bool)uint_option(options, "debug-locate",     init ? 0u         : params.debug.locate);       // debug locate kernel
    params.debug.score      = (bool)uint_option(options, "debug-score",      init ? 1u         : params.debug.score);        // debug score kernel
    params.debug.score_bad  = (bool)uint_option(options, "debug-score-bad",  init ? 0u         : params.debug.score_bad);    // debug score bad
    params.debug.score_info = (bool)uint_option(options, "debug-score-info", init ? 0u         : params.debug.score_info);   // debug score info
    params.debug.reduce     = (bool)uint_option(options, "debug-reduce",     init ? 1u         : params.debug.reduce);       // debug reduce kernel
    params.debug.traceback  = (bool)uint_option(options, "debug-traceback",  init ? 1u         : params.debug.traceback);    // debug traceback kernel
    params.debug.asserts    = (bool)uint_option(options, "debug-asserts",    init ? 1u         : params.debug.asserts);      // debug asserts

    params.persist_batch     =  int_option(options, "persist-batch",         init ? -1         : params.persist_batch);         // persist pass
    params.persist_seeding   =  int_option(options, "persist-seeding",       init ? -1         : params.persist_seeding);       // persist pass
    params.persist_extension =  int_option(options, "persist-extension",     init ? -1         : params.persist_extension);     // persist pass
    params.persist_file      =  string_option(options, "persist-file",       init ? ""         : params.persist_file.c_str() ); // persist file

    params.no_multi_hits     =  int_option(options, "no-multi-hits",      init ? 0          : params.no_multi_hits ); // disable multi-hit selection

    params.max_effort_init = nvbio::max( params.max_effort_init, params.max_effort );
    params.max_ext         = nvbio::max( params.max_ext,         params.max_effort );

    UberScoringScheme& sc = params.scoring_scheme;

    // set the default ED values, or reset them if the scoring mode has been changed
    if (init || (params.scoring_mode != old_scoring_mode))
        sc.ed.m_score_min = SimpleFunc( SimpleFunc::LinearFunc, -(float)params.max_dist, 0.0f );

    // set the default SW values, or reset them if the alignment type has been changed
    if (init || (local != old_local))
    {
        sc.sw = local ? 
            SmithWatermanScoringScheme<>::local() :
            SmithWatermanScoringScheme<>();
    }

    // load scoring scheme from file
    if (params.scoring_file != "")
        sc.sw = load_scoring_scheme( params.scoring_file.c_str(), AlignmentType( params.alignment_type ) );

    // score-min
    sc.ed.m_score_min = func_option( options, "score-min", sc.ed.m_score_min );
    sc.sw.m_score_min = func_option( options, "score-min", sc.sw.m_score_min );

    // match bonus
    sc.sw.m_match.m_val = int_option( options, "ma", sc.sw.m_match.m_val );

    // mismatch penalties
    const int2 mp = int2_option( options, "mp", make_int2( sc.sw.m_mmp.m_max_val, sc.sw.m_mmp.m_min_val ) );
    sc.sw.m_mmp.m_max_val = mp.x;
    sc.sw.m_mmp.m_min_val = mp.y;

    // np
    sc.sw.m_np.m_val = int_option( options, "np", sc.sw.m_np.m_val );

    // read gaps
    const int2 rdg         = int2_option( options, "rdg", make_int2( sc.sw.m_read_gap_const, sc.sw.m_read_gap_coeff ) );
    sc.sw.m_read_gap_const = rdg.x;
    sc.sw.m_read_gap_coeff = rdg.y;

    // reference gaps
    const int2 rfg        = int2_option( options, "rfg", make_int2( sc.sw.m_ref_gap_const, sc.sw.m_ref_gap_coeff ) );
    sc.sw.m_ref_gap_const = rfg.x;
    sc.sw.m_ref_gap_coeff = rfg.y;
}

//
// single-end driver
//
int driver(
    const char*                              output_name, 
    const io::SequenceData&                  reference_data_host,
    const io::FMIndexData&                   driver_data_host,
          io::SequenceDataStream&            read_data_stream,
    const std::map<std::string,std::string>& options,
    const std::string&                       cmdline,
    const std::string&                       rg_id,
    const std::string&                       rg_string)
{
    log_visible(stderr, "Bowtie2 cuda driver... started\n");

    // WARNING: we don't do any error checking on passed parameters!
    Params params;
    {
        bool init = true;
        std::string config = string_option(options, "config", "" );
        if (config != "") { parse_options( params, load_options( config.c_str() ), init ); init = false; }
                            parse_options( params, options,                        init );

    }
    if (params.alignment_type == LocalAlignment &&
        params.scoring_mode == EditDistanceMode)
    {
        log_warning(stderr, "edit-distance scoring is incompatible with local alignment, switching to Smith-Waterman\n");
        params.scoring_mode = SmithWatermanMode;
    }

    // build an empty report
    FILE* html_output = (params.report != std::string("")) ? fopen( params.report.c_str(), "w" ) : NULL;
    if (html_output)
    {
        // encapsulate the document
        {
            html::html_object html( html_output );
            {
                const char* meta_list = "<meta http-equiv=\"refresh\" content=\"1\" />";

                { html::header_object hd( html_output, "Bowtie2 Report", html::style(), meta_list ); }
                { html::body_object body( html_output ); }
            }
        }
        fclose( html_output );
    }

    // compute band length
    const uint32 band_len = Aligner::band_length( params.max_dist );
    const SimpleFunc& score_min = params.scoring_mode == EditDistanceMode ? params.scoring_scheme.ed.m_score_min : params.scoring_scheme.sw.m_score_min;
    
    // print command line options
    log_visible(stderr, "  mode           = %s\n", mapping_mode( params.mode ));
    log_visible(stderr, "  scoring        = %s\n", scoring_mode( params.scoring_mode ));
    log_visible(stderr, "  score-min      = %s:%.2f:%.2f\n", score_min.type_string(), score_min.k, score_min.m);
    log_visible(stderr, "  alignment type = %s\n", params.alignment_type == LocalAlignment ? "local" : "end-to-end");
    log_visible(stderr, "  seed length    = %u\n", params.seed_len);
    log_visible(stderr, "  seed interval  = (%s, %.3f, %.3f)\n", params.seed_freq.type_symbol(), params.seed_freq.k, params.seed_freq.m);
    log_visible(stderr, "  seed rounds    = %u\n", params.max_reseed);
    log_visible(stderr, "  max hits       = %u\n", params.max_hits);
    log_visible(stderr, "  max edit dist  = %u (band len %u)\n", params.max_dist, band_len);
    log_visible(stderr, "  max effort     = %u\n", params.max_effort);
    log_visible(stderr, "  substitutions  = %u\n", params.allow_sub);
    log_visible(stderr, "  mapQ filter    = %u\n", params.mapq_filter);
    log_visible(stderr, "  randomized     = %s\n", params.randomized ? "yes" : "no");
    if (params.allow_sub)
        log_visible(stderr, "  subseed length = %u\n", params.subseed_len);

    const bool need_reverse =
        (params.allow_sub == 0 && USE_REVERSE_INDEX) ||
        (params.allow_sub == 1 && params.subseed_len == 0 && params.mode == BestMappingApprox);

    Timer timer;

    timer.start();

    io::SequenceDataDevice reference_data( reference_data_host );

    io::FMIndexDataDevice driver_data( driver_data_host,
                        io::FMIndexDataDevice::FORWARD |
        (need_reverse ? io::FMIndexDataDevice::REVERSE : 0u) |
                        io::FMIndexDataDevice::SA );

    timer.stop();

    log_stats(stderr, "  allocated device driver data (%.2f GB - %.1fs)\n", float(driver_data.allocated()) / 1.0e9f, timer.seconds() );

    typedef FMIndexDef::type fm_index_type;

    fm_index_type fmi  = driver_data.index();
    fm_index_type rfmi = driver_data.rindex();

    Aligner aligner;

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    log_stats(stderr, "  device has %ld of %ld MB free\n", free/1024/1024, total/1024/1024);

    uint32 BATCH_SIZE;

    for (BATCH_SIZE = params.max_batch_size*1024; BATCH_SIZE >= 16*1024; BATCH_SIZE /= 2)
    {
        // leave some guard band of free memory
        const uint32 guard_band = 600*1024*1024;

        // gauge how much memory we'd need
        const std::pair<uint64,uint64> mem_stats = aligner.init_alloc( BATCH_SIZE, params, kSingleEnd, false );
        if (mem_stats.second < free - guard_band)
            break;
    }
    log_stats(stderr, "  processing reads in batches of %uK\n", BATCH_SIZE/1024);

    if (aligner.init( BATCH_SIZE, params, kSingleEnd ) == false)
        return 1;

    nvbio::cuda::check_error("cuda initializations");

    cudaMemGetInfo(&free, &total);
    log_stats(stderr, "  ready to start processing: device has %ld MB free\n", free/1024/1024);

    float polling_time = 0.0f;
    Timer global_timer;
    global_timer.start();

    UberScoringScheme& scoring_scheme = params.scoring_scheme;

    Stats stats( params );

    aligner.output_file = io::OutputFile::open(output_name,
                                               io::SINGLE_END,
                                               io::BNT(reference_data_host));

    aligner.output_file->set_rg( rg_id.c_str(), rg_string.c_str() );
    aligner.output_file->set_program(
        "nvBowtie",
        "nvBowtie",
        NVBIO_VERSION_STRING,
        cmdline.c_str() );

    aligner.output_file->configure_mapq_evaluator(params.mapq_filter);

    aligner.output_file->header();

    // setup the input thread
    InputThread input_thread( &read_data_stream, stats, BATCH_SIZE );
    input_thread.create();

    uint32 input_set  = 0;
    uint32 n_reads    = 0;

    io::SequenceDataHost local_read_data_host;

    // loop through the batches of reads
    for (uint32 read_begin = 0; true; read_begin += BATCH_SIZE)
    {
        Timer polling_timer;
        polling_timer.start();

        // poll until the current input set is loaded...
        while (input_thread.read_data[ input_set ] == NULL) { yield(); }

        // make sure the other writes are seen
        host_acquire_fence();

        polling_timer.stop();
        polling_time += polling_timer.seconds();

        io::SequenceDataHost* read_data_host = input_thread.read_data[ input_set ];
        if (read_data_host == (io::SequenceDataHost*)InputThread::INVALID)
        {
            log_verbose(stderr, "end of input reached\n");
            break;
        }

        if (read_data_host->max_sequence_len() > Aligner::MAX_READ_LEN)
        {
            log_error(stderr, "unsupported read length %u (maximum is %u)\n",
                read_data_host->max_sequence_len(),
                Aligner::MAX_READ_LEN );
            break;
        }

        // make a local copy of the host batch
        local_read_data_host = *read_data_host;

        // mark this set as ready to be reused
        input_thread.read_data[ input_set ] = NULL;

        // make sure the other threads see this change
        host_release_fence();

        // advance input set pointer
        input_set = (input_set + 1) % InputThread::BUFFERS;

        Timer timer;
        timer.start();

        aligner.output_file->start_batch( &local_read_data_host );

        io::SequenceDataDevice read_data( local_read_data_host );
        cudaThreadSynchronize();

        timer.stop();
        stats.read_HtoD.add( read_data.size(), timer.seconds() );

        const uint32 count = read_data.size();
        log_info(stderr, "aligning reads [%u, %u]\n", read_begin, read_begin + count - 1u);
        log_verbose(stderr, "  %u reads\n", count);
        log_verbose(stderr, "  %.3f M bps (%.1f MB)\n", float(read_data.bps())/1.0e6f, float(read_data.words()*sizeof(uint32)+read_data.bps()*sizeof(char))/float(1024*1024));
        log_verbose(stderr, "  %.1f bps/read (min: %u, max: %u)\n", float(read_data.bps())/float(read_data.size()), read_data.min_sequence_len(), read_data.max_sequence_len());

        if (params.mode == AllMapping)
        {
            if (params.scoring_mode == EditDistanceMode)
            {
                all_ed(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data,
                    stats );
            }
            else
            {
                all_sw(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data,
                    stats );
            }
        }
        else
        {
            if (params.scoring_mode == EditDistanceMode)
            {
                best_approx_ed(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data,
                    stats );
            }
            else
            {
                best_approx_sw(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data,
                    stats );
            }
        }

        global_timer.stop();
        stats.global_time += global_timer.seconds();
        global_timer.start();

        aligner.output_file->end_batch();

        // increase the total reads counter
        n_reads += count;

        log_verbose(stderr, "  %.1f K reads/s\n", 1.0e-3f * float(n_reads) / stats.global_time);
    }

    input_thread.join();

    io::IOStats iostats;

    aligner.output_file->close();

    // transfer I/O statistics to the old stats struct
    iostats = aligner.output_file->get_aggregate_statistics();

    stats.alignments_DtoH.add(iostats.alignments_DtoH_count, iostats.alignments_DtoH_time);
    stats.io = iostats.output_process_timings;
    stats.n_reads           = n_reads;
    stats.n_mapped          = stats.mate1.n_mapped;
    stats.n_ambiguous       = stats.mate1.n_ambiguous;
    stats.n_nonambiguous    = stats.mate1.n_unambiguous;
    stats.n_unique          = stats.mate1.n_unique;
    stats.n_multiple        = stats.mate1.n_multiple;
    stats.mapped            = stats.mate1.mapped_ed_histogram;
    stats.f_mapped          = stats.mate1.mapped_ed_histogram_fwd;
    stats.r_mapped          = stats.mate1.mapped_ed_histogram_rev;
    memcpy(stats.mapq_bins, stats.mate1.mapq_bins,             sizeof(stats.mate1.mapq_bins));
    memcpy(stats.mapped2,   stats.mate1.mapped_ed_correlation, sizeof(stats.mate1.mapped_ed_correlation));

    delete aligner.output_file;

    global_timer.stop();
    stats.global_time += global_timer.seconds();

    nvbio::bowtie2::cuda::generate_report(stats, params.report.c_str());

    log_stats(stderr, "  total        : %.2f sec (avg: %.1fK reads/s).\n", stats.global_time, 1.0e-3f * float(n_reads)/stats.global_time);
    log_stats(stderr, "  mapping      : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.map.time, 1.0e-6f * stats.map.avg_speed(), 1.0e-6f * stats.map.max_speed, stats.map.device_time);
    log_stats(stderr, "  selecting    : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.select.time, 1.0e-6f * stats.select.avg_speed(), 1.0e-6f * stats.select.max_speed, stats.select.device_time);
    log_stats(stderr, "  sorting      : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.sort.time, 1.0e-6f * stats.sort.avg_speed(), 1.0e-6f * stats.sort.max_speed, stats.sort.device_time);
    log_stats(stderr, "  scoring      : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.score.time, 1.0e-6f * stats.score.avg_speed(), 1.0e-6f * stats.score.max_speed, stats.score.device_time);
    log_stats(stderr, "  locating     : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.locate.time, 1.0e-6f * stats.locate.avg_speed(), 1.0e-6f * stats.locate.max_speed, stats.locate.device_time);
    log_stats(stderr, "  backtracking : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.backtrack.time, 1.0e-6f * stats.backtrack.avg_speed(), 1.0e-6f * stats.backtrack.max_speed, stats.backtrack.device_time);
    log_stats(stderr, "  finalizing   : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.finalize.time, 1.0e-6f * stats.finalize.avg_speed(), 1.0e-6f * stats.finalize.max_speed, stats.finalize.device_time);
    log_stats(stderr, "  results DtoH : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.alignments_DtoH.time, 1.0e-6f * stats.alignments_DtoH.avg_speed(), 1.0e-6f * stats.alignments_DtoH.max_speed);
    log_stats(stderr, "  reads HtoD   : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.read_HtoD.time, 1.0e-6f * stats.read_HtoD.avg_speed(), 1.0e-6f * stats.read_HtoD.max_speed);
    log_stats(stderr, "  reads I/O    : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.read_io.time, 1.0e-6f * stats.read_io.avg_speed(), 1.0e-6f * stats.read_io.max_speed);
    log_stats(stderr, "    exposed    : %.2f sec (avg: %.3fK reads/s).\n", polling_time, 1.0e-3f * float(n_reads)/polling_time);
    log_stats(stderr, "  output I/O   : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.io.time, 1.0e-6f * stats.io.avg_speed(), 1.0e-6f * stats.io.max_speed);

    std::vector<uint32>& mapped         = stats.mapped;
    uint32&              n_mapped       = stats.n_mapped;
    uint32&              n_unique       = stats.n_unique;
    uint32&              n_ambiguous    = stats.n_ambiguous;
    uint32&              n_nonambiguous = stats.n_nonambiguous;
    uint32&              n_multiple     = stats.n_multiple;
    {
        log_stats(stderr, "  mapped reads : %.2f %% - of these:\n", 100.0f * float(n_mapped)/float(n_reads) );
        log_stats(stderr, "    aligned uniquely      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_unique)/float(n_mapped), 100.0f * float(n_mapped - n_multiple)/float(n_reads) );
        log_stats(stderr, "    aligned unambiguously : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_nonambiguous)/float(n_mapped), 100.0f * float(n_nonambiguous)/float(n_reads) );
        log_stats(stderr, "    aligned ambiguously   : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_ambiguous)/float(n_mapped), 100.0f * float(n_ambiguous)/float(n_reads) );
        log_stats(stderr, "    aligned multiply      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_multiple)/float(n_mapped), 100.0f * float(n_multiple)/float(n_reads) );
        for (uint32 i = 0; i < mapped.size(); ++i)
        {
            if (float(mapped[i])/float(n_reads) > 1.0e-3f)
                log_stats(stderr, "    ed %4u : %.1f %%\n", i,
                100.0f * float(mapped[i])/float(n_reads) );
        }
    }

    log_visible(stderr, "Bowtie2 cuda driver... done\n");
    return 0;
}

//
// paired-end driver
//
int driver(
    const char*                              output_name, 
    const io::SequenceData&                  reference_data_host,
    const io::FMIndexData&                   driver_data_host,
    const io::PairedEndPolicy                pe_policy,
          io::SequenceDataStream&            read_data_stream1,
          io::SequenceDataStream&            read_data_stream2,
    const std::map<std::string,std::string>& options,
    const std::string&                       cmdline,
    const std::string&                       rg_id,
    const std::string&                       rg_string)
{
    log_visible(stderr, "Bowtie2 cuda driver... started\n");

    // WARNING: we don't do any error checking on passed parameters!
    Params params;
    params.pe_policy = pe_policy;
    {
        bool init = true;
        std::string config = string_option(options, "config", "" );
        if (config != "") { parse_options( params, load_options( config.c_str() ), init ); init = false; }
                            parse_options( params, options,                        init );
    }
    if (params.alignment_type == LocalAlignment &&
        params.scoring_mode == EditDistanceMode)
    {
        log_warning(stderr, "edit-distance scoring is incompatible with local alignment, switching to Smith-Waterman\n");
        params.scoring_mode = SmithWatermanMode;
    }

    // clear the persistance files
    if (params.persist_file != "")
        persist_clear( params.persist_file );

    // build an empty report
    FILE* html_output = (params.report != std::string("")) ? fopen( params.report.c_str(), "w" ) : NULL;
    if (html_output)
    {
        // encapsulate the document
        {
            html::html_object html( html_output );
            {
                const char* meta_list = "<meta http-equiv=\"refresh\" content=\"1\" />";

                { html::header_object hd( html_output, "Bowtie2 Report", html::style(), meta_list ); }
                { html::body_object body( html_output ); }
            }
        }
        fclose( html_output );
    }

    // compute band length
    const uint32 band_len = Aligner::band_length( params.max_dist );
    const SimpleFunc& score_min = EditDistanceMode ? params.scoring_scheme.ed.m_score_min : params.scoring_scheme.sw.m_score_min;

    // print command line options
    log_visible(stderr, "  mode           = %s\n", mapping_mode( params.mode ));
    log_visible(stderr, "  scoring        = %s\n", scoring_mode( params.scoring_mode ));
    log_visible(stderr, "  score-min      = %s:%.2f:%.2f\n", score_min.type_string(), score_min.k, score_min.m);
    log_visible(stderr, "  alignment type = %s\n", params.alignment_type == LocalAlignment ? "local" : "end-to-end");
    log_visible(stderr, "  pe-policy      = %s\n",
                                                   pe_policy == io::PE_POLICY_FF ? "ff" :
                                                   pe_policy == io::PE_POLICY_FR ? "fr" :
                                                   pe_policy == io::PE_POLICY_RF ? "rf" :
                                                                                   "rr" );
    log_visible(stderr, "  seed length    = %u\n", params.seed_len);
    log_visible(stderr, "  seed interval  = (%s, %.3f, %.3f)\n", params.seed_freq.type_symbol(), params.seed_freq.k, params.seed_freq.m);
    log_visible(stderr, "  seed rounds    = %u\n", params.max_reseed);
    log_visible(stderr, "  max hits       = %u\n", params.max_hits);
    log_visible(stderr, "  max edit dist  = %u (band len %u)\n", params.max_dist, band_len);
    log_visible(stderr, "  max effort     = %u\n", params.max_effort);
    log_visible(stderr, "  substitutions  = %u\n", params.allow_sub);
    log_visible(stderr, "  mapQ filter    = %u\n", params.mapq_filter);
    log_visible(stderr, "  randomized     = %s\n", params.randomized ? "yes" : "no");
    if (params.allow_sub)
        log_visible(stderr, "  subseed length = %u\n", params.subseed_len);

    const bool need_reverse =
        (params.allow_sub == 0 && USE_REVERSE_INDEX) ||
        (params.allow_sub == 1 && params.subseed_len == 0 && params.mode == BestMappingApprox);

    io::SequenceDataDevice reference_data( reference_data_host );

    io::FMIndexDataDevice driver_data( driver_data_host,
                        io::FMIndexDataDevice::FORWARD |
        (need_reverse ? io::FMIndexDataDevice::REVERSE : 0u) |
                        io::FMIndexDataDevice::SA );

    log_stats(stderr, "  allocated device driver data (%.2f GB)\n", float(driver_data.allocated()) / 1.0e9f );

    typedef FMIndexDef::type fm_index_type;

    fm_index_type fmi  = driver_data.index();
    fm_index_type rfmi = driver_data.rindex();

    Aligner aligner;

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    log_stats(stderr, "  device has %ld of %ld MB free\n", free/1024/1024, total/1024/1024);

    uint32 BATCH_SIZE;

    for (BATCH_SIZE = params.max_batch_size*1024; BATCH_SIZE >= 16*1024; BATCH_SIZE /= 2)
    {
        // leave some guard band of free memory
        const uint32 guard_band = 600*1024*1024;

        // gauge how much memory we'd need
        const std::pair<uint64,uint64> mem_stats = aligner.init_alloc( BATCH_SIZE, params, kPairedEnds, false );
        if (mem_stats.second < free - guard_band)
            break;
    }
    log_stats(stderr, "  processing reads in batches of %uK\n", BATCH_SIZE/1024);

    if (aligner.init( BATCH_SIZE, params, kPairedEnds ) == false)
        return 1;

    nvbio::cuda::check_error("cuda initializations");

    cudaMemGetInfo(&free, &total);
    log_stats(stderr, "  ready to start processing: device has %ld MB free\n", free/1024/1024);

    size_t stack_size_limit;
    cudaDeviceGetLimit( &stack_size_limit, cudaLimitStackSize );
    log_debug(stderr, "    max cuda stack size: %u\n", stack_size_limit);

    float polling_time = 0.0f;
    Timer timer;
    Timer global_timer;
    global_timer.start();

    UberScoringScheme& scoring_scheme = params.scoring_scheme;

    Stats stats( params );

    aligner.output_file = io::OutputFile::open(output_name,
                                               io::PAIRED_END,
                                               io::BNT(reference_data_host));

    aligner.output_file->set_rg( rg_id.c_str(), rg_string.c_str() );
    aligner.output_file->set_program(
        "nvBowtie",
        "nvBowtie",
        NVBIO_VERSION_STRING,
        cmdline.c_str() );

    aligner.output_file->configure_mapq_evaluator(params.mapq_filter);

    aligner.output_file->header();

    // setup the input thread
    InputThreadPaired input_thread( &read_data_stream1, &read_data_stream2, stats, BATCH_SIZE );
    input_thread.create();

    uint32 input_set  = 0;
    uint32 n_reads    = 0;

    io::SequenceDataHost local_read_data_host1;
    io::SequenceDataHost local_read_data_host2;

    // loop through the batches of reads
    for (uint32 read_begin = 0; true; read_begin += BATCH_SIZE)
    {
        Timer polling_timer;
        polling_timer.start();

        // poll until the current input set is loaded...
        while (input_thread.read_data1[ input_set ] == NULL ||
               input_thread.read_data2[ input_set ] == NULL) { yield(); }

        // make sure the other writes are seen
        host_acquire_fence();

        polling_timer.stop();
        polling_time += polling_timer.seconds();

        io::SequenceDataHost* read_data_host1 = input_thread.read_data1[ input_set ];
        io::SequenceDataHost* read_data_host2 = input_thread.read_data2[ input_set ];
        if (read_data_host1 == (io::SequenceDataHost*)InputThread::INVALID ||
            read_data_host2 == (io::SequenceDataHost*)InputThread::INVALID)
        {
            log_verbose(stderr, "end of input reached\n");
            break;
        }

        if ((read_data_host1->max_sequence_len() > Aligner::MAX_READ_LEN) ||
            (read_data_host2->max_sequence_len() > Aligner::MAX_READ_LEN))
        {
            log_error(stderr, "unsupported read length %u (maximum is %u)\n",
                nvbio::max(read_data_host1->max_sequence_len(), read_data_host2->max_sequence_len()),
                Aligner::MAX_READ_LEN );
            break;
        }

        // make a local copy of the host batch
        local_read_data_host1 = *read_data_host1;
        local_read_data_host2 = *read_data_host2;

        // mark this set as ready to be reused
        input_thread.read_data1[ input_set ] = NULL;
        input_thread.read_data2[ input_set ] = NULL;

        // make sure the other threads see this change
        host_release_fence();

        // advance input set pointer
        input_set = (input_set + 1) % InputThread::BUFFERS;

        Timer timer;
        timer.start();

        aligner.output_file->start_batch( &local_read_data_host1, &local_read_data_host2 );

        io::SequenceDataDevice read_data1( local_read_data_host1/*, io::ReadDataDevice::READS | io::ReadDataDevice::QUALS*/ );
        io::SequenceDataDevice read_data2( local_read_data_host2/*, io::ReadDataDevice::READS | io::ReadDataDevice::QUALS*/ );

        timer.stop();
        stats.read_HtoD.add( read_data1.size(), timer.seconds() );

        const uint32 count = read_data1.size();
        log_info(stderr, "aligning reads [%u, %u]\n", read_begin, read_begin + count - 1u);
        log_verbose(stderr, "  %u reads\n", count);
        log_verbose(stderr, "  %.3f M bps (%.1f MB)\n",
            float(read_data1.bps() + read_data2.bps())/1.0e6f,
            float(read_data1.words()*sizeof(uint32)+read_data1.bps()*sizeof(char))/float(1024*1024)+
            float(read_data2.words()*sizeof(uint32)+read_data2.bps()*sizeof(char))/float(1024*1024));
        log_verbose(stderr, "  %.1f bps/read (min: %u, max: %u)\n",
            float(read_data1.bps()+read_data2.bps())/float(read_data1.size()+read_data2.size()),
            nvbio::min( read_data1.min_sequence_len(), read_data2.min_sequence_len() ),
            nvbio::max( read_data1.max_sequence_len(), read_data2.max_sequence_len() ));

        if (params.mode == AllMapping)
        {
            log_error(stderr, "paired-end all-mapping is not yet supported!\n");
            exit(1);
        }
        else
        {
            if (params.scoring_mode == EditDistanceMode)
            {
                best_approx_ed(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data1,
                    read_data2,
                    stats );
            }
            else
            {
                best_approx_sw(
                    aligner,
                    params,
                    fmi,
                    rfmi,
                    scoring_scheme,
                    reference_data,
                    driver_data,
                    read_data1,
                    read_data2,
                    stats );
            }
        }

        global_timer.stop();
        stats.global_time += global_timer.seconds();
        global_timer.start();

        aligner.output_file->end_batch();

        // increase the total reads counter
        n_reads += count;

        log_verbose(stderr, "  %.1f K reads/s\n", 1.0e-3f * float(n_reads) / stats.global_time);
    }

    input_thread.join();

    io::IOStats iostats;

    aligner.output_file->close();

    // transfer I/O statistics
    iostats = aligner.output_file->get_aggregate_statistics();

    stats.alignments_DtoH.add(iostats.alignments_DtoH_count, iostats.alignments_DtoH_time);
    stats.io                = iostats.output_process_timings;
    stats.n_reads           = n_reads;
    stats.n_mapped          = stats.paired.n_mapped;
    stats.n_ambiguous       = stats.paired.n_ambiguous;
    stats.n_nonambiguous    = stats.paired.n_unambiguous;
    stats.n_unique          = stats.paired.n_unique;
    stats.n_multiple        = stats.paired.n_multiple;
    stats.mapped            = stats.paired.mapped_ed_histogram;
    stats.f_mapped          = stats.paired.mapped_ed_histogram_fwd;
    stats.r_mapped          = stats.paired.mapped_ed_histogram_rev;
    memcpy(stats.mapq_bins, stats.paired.mapq_bins,             sizeof(stats.paired.mapq_bins));
    memcpy(stats.mapped2,   stats.paired.mapped_ed_correlation, sizeof(stats.paired.mapped_ed_correlation));

    delete aligner.output_file;

    global_timer.stop();
    stats.global_time += global_timer.seconds();

    nvbio::bowtie2::cuda::generate_report(stats, params.report.c_str());

    log_stats(stderr, "  total          : %.2f sec (avg: %.1fK reads/s).\n", stats.global_time, 1.0e-3f * float(n_reads)/stats.global_time);
    log_stats(stderr, "  mapping        : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.map.time, 1.0e-6f * stats.map.avg_speed(), 1.0e-6f * stats.map.max_speed, stats.map.device_time);
    log_stats(stderr, "  scoring        : %.2f sec (avg: %.1fM reads/s, max: %.3fM reads/s, %.2f device sec).).\n", stats.scoring_pipe.time, 1.0e-6f * stats.scoring_pipe.avg_speed(), 1.0e-6f * stats.scoring_pipe.max_speed, stats.scoring_pipe.device_time);
    log_stats(stderr, "    selecting    : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.select.time, 1.0e-6f * stats.select.avg_speed(), 1.0e-6f * stats.select.max_speed, stats.select.device_time);
    log_stats(stderr, "    sorting      : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.sort.time, 1.0e-6f * stats.sort.avg_speed(), 1.0e-6f * stats.sort.max_speed, stats.sort.device_time);
    log_stats(stderr, "    scoring(a)   : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.score.time, 1.0e-6f * stats.score.avg_speed(), 1.0e-6f * stats.score.max_speed, stats.score.device_time);
    log_stats(stderr, "    scoring(o)   : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.opposite_score.time, 1.0e-6f * stats.opposite_score.avg_speed(), 1.0e-6f * stats.opposite_score.max_speed, stats.opposite_score.device_time);
    log_stats(stderr, "    locating     : %.2f sec (avg: %.3fM seeds/s, max: %.3fM seeds/s, %.2f device sec).\n", stats.locate.time, 1.0e-6f * stats.locate.avg_speed(), 1.0e-6f * stats.locate.max_speed, stats.locate.device_time);
    log_stats(stderr, "  backtracing(a) : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.backtrack.time, 1.0e-6f * stats.backtrack.avg_speed(), 1.0e-6f * stats.backtrack.max_speed, stats.backtrack.device_time);
    log_stats(stderr, "  backtracing(o) : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.backtrack_opposite.time, 1.0e-6f * stats.backtrack_opposite.avg_speed(), 1.0e-6f * stats.backtrack_opposite.max_speed, stats.backtrack_opposite.device_time);
    log_stats(stderr, "  finalizing     : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s, %.2f device sec).\n", stats.finalize.time, 1.0e-6f * stats.finalize.avg_speed(), 1.0e-6f * stats.finalize.max_speed, stats.finalize.device_time);
    log_stats(stderr, "  results DtoH   : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.alignments_DtoH.time, 1.0e-6f * stats.alignments_DtoH.avg_speed(), 1.0e-6f * stats.alignments_DtoH.max_speed);
    log_stats(stderr, "  reads HtoD     : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.read_HtoD.time, 1.0e-6f * stats.read_HtoD.avg_speed(), 1.0e-6f * stats.read_HtoD.max_speed);
    log_stats(stderr, "  reads I/O      : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.read_io.time, 1.0e-6f * stats.read_io.avg_speed(), 1.0e-6f * stats.read_io.max_speed);
    log_stats(stderr, "    exposed      : %.2f sec (avg: %.3fK reads/s).\n", polling_time, 1.0e-3f * float(n_reads)/polling_time);
    log_stats(stderr, "  output I/O     : %.2f sec (avg: %.3fM reads/s, max: %.3fM reads/s).\n", stats.io.time, 1.0e-6f * stats.io.avg_speed(), 1.0e-6f * stats.io.max_speed);

    std::vector<uint32>& mapped         = stats.mapped;
    uint32&              n_mapped       = stats.n_mapped;
    uint32&              n_unique       = stats.n_unique;
    uint32&              n_ambiguous    = stats.n_ambiguous;
    uint32&              n_nonambiguous = stats.n_nonambiguous;
    uint32&              n_multiple     = stats.n_multiple;
    {
        log_stats(stderr, "  concordant reads : %.2f %% - of these:\n", 100.0f * float(n_mapped)/float(n_reads) );
        log_stats(stderr, "    aligned uniquely      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_unique)/float(n_mapped), 100.0f * float(n_mapped - n_multiple)/float(n_reads) );
        log_stats(stderr, "    aligned unambiguously : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_nonambiguous)/float(n_mapped), 100.0f * float(n_nonambiguous)/float(n_reads) );
        log_stats(stderr, "    aligned ambiguously   : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_ambiguous)/float(n_mapped), 100.0f * float(n_ambiguous)/float(n_reads) );
        log_stats(stderr, "    aligned multiply      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(n_multiple)/float(n_mapped), 100.0f * float(n_multiple)/float(n_reads) );
        for (uint32 i = 0; i < mapped.size(); ++i)
        {
            if (float(mapped[i])/float(n_reads) > 1.0e-3f)
                log_stats(stderr, "    ed %4u : %.1f %%\n", i,
                100.0f * float(mapped[i])/float(n_reads) );
        }

        log_stats(stderr, "  mate1 : %.2f %% - of these:\n", 100.0f * float(stats.mate1.n_mapped)/float(n_reads) );
        if (stats.mate1.n_mapped)
        {
            log_stats(stderr, "    aligned uniquely      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate1.n_unique)/float(stats.mate1.n_mapped), 100.0f * float(stats.mate1.n_mapped - stats.mate1.n_multiple)/float(n_reads) );
            log_stats(stderr, "    aligned unambiguously : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate1.n_unambiguous)/float(stats.mate1.n_mapped), 100.0f * float(stats.mate1.n_unambiguous)/float(n_reads) );
            log_stats(stderr, "    aligned ambiguously   : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate1.n_ambiguous)/float(stats.mate1.n_mapped), 100.0f * float(stats.mate1.n_ambiguous)/float(n_reads) );
            log_stats(stderr, "    aligned multiply      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate1.n_multiple)/float(stats.mate1.n_mapped), 100.0f * float(stats.mate1.n_multiple)/float(n_reads) );
        }

        log_stats(stderr, "  mate2 : %.2f %% - of these:\n", 100.0f * float(stats.mate2.n_mapped)/float(n_reads) );
        if (stats.mate2.n_mapped)
        {
            log_stats(stderr, "    aligned uniquely      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate2.n_unique)/float(stats.mate2.n_mapped), 100.0f * float(stats.mate2.n_mapped - stats.mate2.n_multiple)/float(n_reads) );
            log_stats(stderr, "    aligned unambiguously : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate2.n_unambiguous)/float(stats.mate2.n_mapped), 100.0f * float(stats.mate2.n_unambiguous)/float(n_reads) );
            log_stats(stderr, "    aligned ambiguously   : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate2.n_ambiguous)/float(stats.mate2.n_mapped), 100.0f * float(stats.mate2.n_ambiguous)/float(n_reads) );
            log_stats(stderr, "    aligned multiply      : %4.1f%% (%4.1f%% of total)\n", 100.0f * float(stats.mate2.n_multiple)/float(stats.mate2.n_mapped), 100.0f * float(stats.mate2.n_multiple)/float(n_reads) );
        }
    }

    log_visible(stderr, "Bowtie2 cuda driver... done\n");
    return 0;
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
