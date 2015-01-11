#include <nvBowtie/bowtie2/cuda/params.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <numeric>
#include <functional>
#include <map>

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
    params.keep_stats       = uint_option(options, "stats",                 init ? 1u      : params.keep_stats) ? true : false; // keep stats
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
    params.randomized       = uint_option(options, "rand",                  init ? 1u      : params.randomized) ? true : false; // use randomized selection
    params.randomized       =!uint_option(options, "no-rand",                               !params.randomized) ? true : false; // don't use randomized selection
    params.top_seed         = uint_option(options, "top",                   init ? 0u      : params.top_seed);             // explore top seed entirely
    params.min_read_len     = uint_option(options, "min-read-len",          init ? 12u     : params.min_read_len);         // minimum read length
    params.ungapped_mates   = uint_option(options, "ungapped-mates", "ug",  init ? 0u      : params.ungapped_mates) ? true : false; // ungapped mate alignment

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
        params.seed_freq = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, (local ? 0.75f : 1.15f) );
    }

    params.seed_len         = uint_option(options,  "seed-len",      "L",                    params.seed_len);      // no greater than 32
    params.seed_freq        = func_option( options, "seed-freq",     "i",                    params.seed_freq );    // seed interval
    params.subseed_len      = uint_option(options,  "subseed-len",          init ? 0u      : params.subseed_len);   // no greater than 32

    params.pe_overlap    =  uint_option(options, "overlap",         init ? 1u      : params.pe_overlap) ? true : false;            // paired-end overlap
    params.pe_overlap    = !uint_option(options, "no-overlap",                      !params.pe_overlap) ? true : false;            // paired-end overlap
    params.pe_dovetail   =  uint_option(options, "dovetail",        init ? 0u      : params.pe_dovetail) ? true : false;           // paired-end dovetail
    params.pe_unpaired   = !uint_option(options, "no-mixed",        init ? 0u      :!params.pe_unpaired) ? true : false;           // paired-end no-mixed
    params.min_frag_len  = uint_option(options, "minins", "I",      init ? 0u      : params.min_frag_len);          // paired-end minimum fragment length
    params.max_frag_len  = uint_option(options, "maxins", "X",      init ? 500u    : params.max_frag_len);          // paired-end maximum fragment length

    // the maximum batch of reads processed in parallel
    params.max_batch_size = uint_option(options, "batch-size",  init ? 1024u : params.max_batch_size );   // maximum batch size

    // internal controls
    params.scoring_window   = uint_option(options, "scoring-window",   init ? 32u        : params.scoring_window);       // scoring window size
    params.debug.read_id    = (uint32)int_option(options, "debug-read",      init ? -1         : (int32)params.debug.read_id); // debug read id
    params.debug.select     = uint_option(options, "debug-select",     init ? 0u         : params.debug.select)     ? true : false;       // debug select kernel
    params.debug.locate     = uint_option(options, "debug-locate",     init ? 0u         : params.debug.locate)     ? true : false;       // debug locate kernel
    params.debug.score      = uint_option(options, "debug-score",      init ? 1u         : params.debug.score)      ? true : false;        // debug score kernel
    params.debug.score_bad  = uint_option(options, "debug-score-bad",  init ? 0u         : params.debug.score_bad)  ? true : false;    // debug score bad
    params.debug.score_info = uint_option(options, "debug-score-info", init ? 0u         : params.debug.score_info) ? true : false;   // debug score info
    params.debug.reduce     = uint_option(options, "debug-reduce",     init ? 1u         : params.debug.reduce)     ? true : false;       // debug reduce kernel
    params.debug.traceback  = uint_option(options, "debug-traceback",  init ? 1u         : params.debug.traceback)  ? true : false;    // debug traceback kernel
    params.debug.asserts    = uint_option(options, "debug-asserts",    init ? 1u         : params.debug.asserts)    ? true : false;      // debug asserts

    params.persist_batch     =  int_option(options, "persist-batch",         init ? -1         : params.persist_batch);         // persist pass
    params.persist_seeding   =  int_option(options, "persist-seeding",       init ? -1         : params.persist_seeding);       // persist pass
    params.persist_extension =  int_option(options, "persist-extension",     init ? -1         : params.persist_extension);     // persist pass
    params.persist_file      =  string_option(options, "persist-file",       init ? ""         : params.persist_file.c_str() ); // persist file

    params.no_multi_hits     =  int_option(options, "no-multi-hits",  init ? 0           : params.no_multi_hits ); // disable multi-hit selection

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

    // presets
    if (params.alignment_type == EndToEndAlignment)
    {
        if (uint_option(options, "very-fast", 0u))
        {
            params.max_effort = 5u;
            params.max_reseed = 1u;
            params.seed_len   = 22u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 0.0f, 2.5f );
        }
        if (uint_option(options, "fast", 0u))
        {
            params.max_effort = 10u;
            params.max_reseed = 2u;
            params.seed_len   = 22u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 0.0f, 2.5f );
        }
        if (uint_option(options, "sensitive", 0u))
        {
            params.max_effort = 15u;
            params.max_reseed = 2u;
            params.seed_len   = 22u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 1.15f );
        }
        if (uint_option(options, "very-sensitive", 0u))
        {
            params.max_effort = 20u;
            params.max_reseed = 3u;
            params.seed_len   = 20u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 0.5f );
        }
    }
    else
    {
        if (uint_option(options, "very-fast", "very-fast-local", 0u))
        {
            params.max_effort = 5u;
            params.max_reseed = 1u;
            params.seed_len   = 25u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 2.0f );
        }
        if (uint_option(options, "fast", "fast-local", 0u))
        {
            params.max_effort = 10u;
            params.max_reseed = 2u;
            params.seed_len   = 22u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 1.75f );
        }
        if (uint_option(options, "sensitive", "sensitive-local", 0u))
        {
            params.max_effort = 15u;
            params.max_reseed = 2u;
            params.seed_len   = 20u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 0.75f );
        }
        if (uint_option(options, "very-sensitive", "very-sensitive-local", 0u))
        {
            params.max_effort = 20u;
            params.max_reseed = 3u;
            params.seed_len   = 20u;
            params.seed_freq  = SimpleFunc( SimpleFunc::SqrtFunc, 1.0f, 0.5f );
        }
    }
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
