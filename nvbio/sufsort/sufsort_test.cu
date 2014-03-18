#define NVBIO_CUDA_DEBUG

#include <cub/cub.cuh>

#include <sufsort/sufsort.h>
#include <sufsort/sufsort_utils.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/string_set.h>
#include <nvbio/basic/cuda/ldg.h>
#include <nvbio/io/fmi.h>
#include <nvbio/fmindex/dna.h>
#include <nvbio/fmindex/bwt.h>
#include <thrust/device_vector.h>
#include <omp.h>

// crc init
void crcInit();

namespace nvbio {

namespace sufsort {

template <uint32 SYMBOL_SIZE, typename offset_type>
void make_test_string_set(
    const uint64                        N_strings,
    const uint32                        N,
    thrust::host_vector<uint32>&        h_string,
    thrust::host_vector<offset_type>&   h_offsets)
{
    for (uint64 i = 0; i < N_strings; ++i)
        h_offsets[i] = offset_type( uint64(N)*i );

    LCG_random rand;
    for (uint64 i = 0; i < h_string.size(); ++i)
        h_string[i] = rand.next();

    h_offsets[N_strings] = N*N_strings;
}

struct SuffixHandler
{
    void process(
        const uint32  n_suffixes,
        const uint32* suffix_array,
        const uint32* string_ids,
        const uint32* cum_lengths)
    {
        output.resize( n_suffixes );
        thrust::copy(
            thrust::device_ptr<const uint32>( suffix_array ),
            thrust::device_ptr<const uint32>( suffix_array ) + n_suffixes,
            output.begin() );
    }

    thrust::device_vector<uint32> output;
};

} // namespace sufsort

int sufsort_test(int argc, char* argv[])
{
    enum Test
    {
        kGPU_SA             = 1u,
        kGPU_BWT            = 2u,
        kCPU_BWT            = 4u,
        kGPU_BWT_FUNCTIONAL = 8u,
        kGPU_BWT_GENOME     = 16u,
        kGPU_BWT_SET        = 32u,
        kCPU_BWT_SET        = 64u,
        kGPU_SA_SET         = 128u,
    };
    uint32 TEST_MASK = 0xFFFFFFFFu;

    uint32 gpu_bwt_size = 50u;
    uint32 cpu_bwt_size = 100u;
    uint32 threads      = omp_get_num_procs();
    bool   store_output = true;

    char*  index_name = "data/human.NCBI36/Homo_sapiens.NCBI36.53.dna.toplevel.fa";

    BWTParams params;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-cpu-mem" ) == 0)
        {
            params.host_memory = atoi( argv[++i] ) * uint64(1024u*1024u);
        }
        else if (strcmp( argv[i], "-gpu-mem" ) == 0)
        {
            params.device_memory = atoi( argv[++i] ) * uint64(1024u*1024u);
        }
        else if (strcmp( argv[i], "-cpu-bwt-size" ) == 0)
        {
            cpu_bwt_size = atoi( argv[++i] );
        }
        else if (strcmp( argv[i], "-gpu-bwt-size" ) == 0)
        {
            gpu_bwt_size = atoi( argv[++i] );
        }
        else if (strcmp( argv[i], "-threads" ) == 0)
        {
            threads = atoi( argv[++i] );
        }
        else if (strcmp( argv[i], "-no-output" ) == 0)
        {
            store_output = false;
        }
        else if (strcmp( argv[i], "-tests" ) == 0)
        {
            const std::string tests_string( argv[++i] );

            char temp[256];
            const char* begin = tests_string.c_str();
            const char* end   = begin;

            TEST_MASK = 0u;

            while (1)
            {
                while (*end != ':' && *end != '\0')
                {
                    temp[end - begin] = *end;
                    end++;
                }

                temp[end - begin] = '\0';

                if (strcmp( temp, "gpu-sa" ) == 0)
                    TEST_MASK |= kGPU_SA;
                else if (strcmp( temp, "gpu-bwt" ) == 0)
                    TEST_MASK |= kGPU_BWT;
                else if (strcmp( temp, "gpu-bwt-func" ) == 0)
                    TEST_MASK |= kGPU_BWT_FUNCTIONAL;
                else if (strcmp( temp, "gpu-bwt-genome" ) == 0)
                    TEST_MASK |= kGPU_BWT_GENOME;
                else if (strcmp( temp, "cpu-bwt" ) == 0)
                    TEST_MASK |= kCPU_BWT;
                else if (strcmp( temp, "gpu-set-bwt" ) == 0)
                    TEST_MASK |= kGPU_BWT_SET;
                else if (strcmp( temp, "cpu-set-bwt" ) == 0)
                    TEST_MASK |= kCPU_BWT_SET;

                if (*end == '\0')
                    break;

                ++end; begin = end;
            }
        }
    }

    // Now set the number of threads
    omp_set_num_threads( threads );

    fprintf(stderr, "nvbio/sufsort test... started (%u threads)\n", threads);
    #pragma omp parallel
    {
        fprintf(stderr, "  running on multiple threads\n");
    }
    const uint32 N           = 100;
    const uint32 SYMBOL_SIZE = 2;
    const uint32 SYMBOLS_PER_WORD = (8u*sizeof(uint32)) / SYMBOL_SIZE;

    if (0)
    {
        fprintf(stderr, "\nread \"sort.dat\"... started\n");
        FILE* file = fopen("./sort.dat", "rb");
        uint32 n_active_strings;
        fread( &n_active_strings, sizeof(uint32), 1u, file );
        thrust::host_vector<uint32> h_keys( n_active_strings+4 );
        thrust::host_vector<uint32> h_indices( n_active_strings+4 );
        thrust::host_vector<uint8>  h_temp_flags( n_active_strings+32 );
        fread( nvbio::plain_view( h_keys ),       sizeof(uint32), n_active_strings, file );
        fread( nvbio::plain_view( h_indices ),    sizeof(uint32), n_active_strings, file );
        fread( nvbio::plain_view( h_temp_flags ), sizeof(uint32), (n_active_strings + 31)/32, file );
        fclose( file );
        fprintf(stderr, "read \"sort.dat\"... done: %u entries\n", n_active_strings);

        int current_device;
        cudaGetDevice( &current_device );
        mgpu::ContextPtr mgpu_ctxt = mgpu::CreateCudaDevice( current_device );

        thrust::device_vector<uint32> d_keys( h_keys );
        thrust::device_vector<uint32> d_indices( h_indices );
        thrust::device_vector<uint8>  d_temp_flags( h_temp_flags );

        uint32* d_comp_flags = (uint32*)nvbio::device_view( d_temp_flags );

        // sort within segments
        mgpu::SegSortPairsFromFlags(
            nvbio::device_view( d_keys ),
            nvbio::device_view( d_indices ),
            d_comp_flags,
            n_active_strings,
            *mgpu_ctxt );

        NVBIO_CUDA_DEBUG_STATEMENT( cudaDeviceSynchronize() );
        cuda::check_error("CompressionSort::sort() : seg_sort");
    }

    if (TEST_MASK & kGPU_SA)
    {
        typedef uint32                                                  index_type;
        typedef PackedStream<uint32*,uint8,SYMBOL_SIZE,true,index_type> packed_stream_type;
        typedef packed_stream_type::iterator                            packed_stream_iterator;

        const index_type N_symbols  = 8u*1024u*1024u;
        const index_type N_words    = (N_symbols + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

        fprintf(stderr, "  gpu sa test\n");
        fprintf(stderr, "    %5.1f M symbols\n",  (1.0e-6f*float(N_symbols)));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        thrust::host_vector<uint32> h_string( N_words );

        LCG_random rand;
        for (index_type i = 0; i < N_words; ++i)
            h_string[i] = rand.next();

        for (uint32 lcp = 100; lcp <= 100000; lcp *= 10)
        {
            // insert some long common prefixes
            for (uint32 i = 50; i < 50 + lcp; ++i)
                h_string[i] = 0;

            thrust::device_vector<uint32>  d_string( h_string );
            thrust::device_vector<uint32>  d_sa( N_symbols+1 );

            cudaDeviceSynchronize();

            packed_stream_type d_packed_string( nvbio::plain_view( d_string ) );

            fprintf(stderr, "\n  sa... started (LCP: %u)\n", lcp*16u);

            Timer timer;
            timer.start();

            cuda::suffix_sort(
                N_symbols,
                d_packed_string.begin(),
                d_sa.begin(),
                &params );

            cudaDeviceSynchronize();
            timer.stop();

            fprintf(stderr, "  sa... done: %.2fs (%.1fM suffixes/s)\n", timer.seconds(), 1.0e-6f*float(N_symbols)/float(timer.seconds()));

            if (1)
            {
                fprintf(stderr, "  sa-is... started\n");
                timer.start();

                std::vector<int32> sa_ref( N_symbols+1 );
                gen_sa( N_symbols, packed_stream_type( nvbio::plain_view( h_string ) ), &sa_ref[0] );

                timer.stop();
                fprintf(stderr, "  sa-is... done: %.2fs (%.1fM suffixes/s)\n", timer.seconds(), 1.0e-6f*float(N_symbols)/float(timer.seconds()));

                thrust::host_vector<uint32> h_sa( d_sa );
                for (uint32 i = 0; i < N_symbols; ++i)
                {
                    const uint32 s = h_sa[i];
                    const uint32 r = sa_ref[i];
                    if (s != r)
                    {
                        log_error(stderr, "  mismatch at %u: expected %u, got %u\n", i, r, s);
                        return 0u;
                    }
                }
            }
        }

        FILE* file = fopen("./data/howto", "r" );
        if (file == NULL)
            log_warning(stderr, "  unable to open \"howto\" file\n");
        else
        {
            fprintf(stderr, "\n  loading \"howto\" text benchmark\n");
            fseek( file, 0, SEEK_END );
            const uint32 N_symbols = uint32( ftell( file ) );
            thrust::host_vector<uint8> h_text( N_symbols );
            rewind( file );
            fread( &h_text[0], 1, N_symbols, file );
            fclose( file );

            thrust::device_vector<uint8>   d_text( h_text );
            thrust::device_vector<uint32>  d_sa( N_symbols+1 );

            cudaDeviceSynchronize();

            fprintf(stderr, "  sa... started (%u bytes)\n", N_symbols);

            Timer timer;
            timer.start();

            cuda::suffix_sort(
                N_symbols,
                d_text.begin(),
                d_sa.begin(),
                &params );

            cudaDeviceSynchronize();
            timer.stop();

            fprintf(stderr, "  sa... done: %.2fs (%.1fM suffixes/s)\n", timer.seconds(), 1.0e-6f*float(N_symbols)/float(timer.seconds()));

            if (1)
            {
                fprintf(stderr, "  sa-is... started\n");
                timer.start();

                std::vector<int32> sa_ref( N_symbols+1 );
                sa_ref[0] = N_symbols;
                saisxx( nvbio::plain_view( h_text ), &sa_ref[0] + 1, int32(N_symbols), 256 );

                timer.stop();
                fprintf(stderr, "  sa-is... done: %.2fs (%.1fM suffixes/s)\n", timer.seconds(), 1.0e-6f*float(N_symbols)/float(timer.seconds()));

                thrust::host_vector<uint32> h_sa( d_sa );
                for (uint32 i = 0; i < N_symbols; ++i)
                {
                    const uint32 s = h_sa[i];
                    const uint32 r = sa_ref[i];
                    if (s != r)
                    {
                        log_error(stderr, "  mismatch at %u: expected %u, got %u\n", i, r, s);
                        return 0u;
                    }
                }
            }
        }
    }
    if (TEST_MASK & kGPU_SA_SET)
    {
        typedef PackedStream<uint32*,uint8,SYMBOL_SIZE,false>           packed_stream_type;
        typedef packed_stream_type::iterator                            packed_stream_iterator;
        typedef ConcatenatedStringSet<packed_stream_iterator,uint32*>   string_set;

        const uint32 N_strings  = 1024*1024;
        const uint32 N_tests    = 10;
        const uint32 N_words    = uint32((uint64(N_strings)*N + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD);

        thrust::host_vector<uint32>  h_string( N_words );
        thrust::host_vector<uint32>  h_offsets( N_strings+1 );

        sufsort::make_test_string_set<SYMBOL_SIZE>(
            N_strings,
            N,
            h_string,
            h_offsets );

        thrust::device_vector<uint32>  d_string( h_string );
        thrust::device_vector<uint32>  d_offsets( h_offsets );

        packed_stream_type d_packed_string( nvbio::plain_view( d_string ) );

        string_set d_string_set(
            N_strings,
            d_packed_string.begin(),
            nvbio::plain_view( d_offsets ) );

        cudaDeviceSynchronize();

        fprintf(stderr, "  gpu SA test\n");
        fprintf(stderr, "    %5.1f M strings\n",  (1.0e-6f*float(N_strings)));
        fprintf(stderr, "    %5.1f M suffixes\n", (1.0e-6f*float(N_strings*(N+1))));
        fprintf(stderr, "    %5.1f G symbols\n",  (1.0e-9f*float(uint64(N_strings)*(N+1)*(N+1)/2)));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        // copy a sparse string set into a packed concatenated one
        {
            sufsort::SuffixHandler suffix_hander;

            Timer timer;
            timer.start();

            // sort the suffixes
            for (uint32 i = 0; i < N_tests; ++i)
                cuda::suffix_sort( d_string_set, suffix_hander, &params );

            cudaDeviceSynchronize();
            timer.stop();

            fprintf(stderr, "  sorting time: %.2fs\n", timer.seconds()/float(N_tests));
            fprintf(stderr, "    %5.1f M strings/s\n",  (1.0e-6f*float(N_strings))               * (float(N_tests)/timer.seconds()));
            fprintf(stderr, "    %5.1f M suffixes/s\n", (1.0e-6f*float(N_strings*(N+1)))         * (float(N_tests)/timer.seconds()));
            fprintf(stderr, "    %5.1f G symbols/s\n",  (1.0e-9f*float(uint64(N_strings)*(N+1)*(N+1)/2)) * (float(N_tests)/timer.seconds()));
        }
    }
    if (TEST_MASK & kGPU_BWT_FUNCTIONAL)
    {
        typedef PackedStream<uint32*,uint8,SYMBOL_SIZE,true,uint32>     packed_stream_type;
        typedef packed_stream_type::iterator                            packed_stream_iterator;

        const uint32 N_words    = 8;
        const uint32 N_symbols  = N_words * SYMBOLS_PER_WORD - 13u;

        char char_string[N_symbols+1];

        fprintf(stderr, "  gpu bwt test\n");

        thrust::host_vector<uint32>  h_string( N_words );
        thrust::host_vector<uint32>  h_bwt( N_words+1 );
        thrust::host_vector<uint32>  h_bwt_ref( N_words+1 );
        uint32                       primary_ref;

        LCG_random rand;
        for (uint32 i = 0; i < N_words; ++i)
            h_string[i] = rand.next();

        dna_to_string(
            packed_stream_type( nvbio::plain_view( h_string ) ),
            N_symbols,
            char_string );

        fprintf(stderr, "    str     : %s\n", char_string );
        {
            // generate the SA using SA-IS
            int32 sa[N_symbols+1];
            gen_sa( N_symbols, packed_stream_type( nvbio::plain_view( h_string ) ), &sa[0] );

            // generate the BWT from the SA
            primary_ref = gen_bwt_from_sa( N_symbols, packed_stream_type( nvbio::plain_view( h_string ) ), sa, packed_stream_type( nvbio::plain_view( h_bwt_ref ) ) );

            dna_to_string(
                packed_stream_type( nvbio::plain_view( h_bwt_ref ) ),
                N_symbols,
                char_string );

            fprintf(stderr, "    primary : %u\n", primary_ref );
            fprintf(stderr, "    bwt     : %s\n", char_string );
        }

        thrust::device_vector<uint32>  d_string( h_string );
        thrust::device_vector<uint32>  d_bwt( N_words+1 );

        cudaDeviceSynchronize();

        packed_stream_type d_packed_string( nvbio::plain_view( d_string ) );
        packed_stream_type d_packed_bwt( nvbio::plain_view( d_bwt ) );

        fprintf(stderr, "  bwt... started\n");

        Timer timer;
        timer.start();

        const uint32 primary = cuda::bwt(
            N_symbols,
            d_packed_string.begin(),
            d_packed_bwt.begin(),
            &params );

        timer.stop();

        fprintf(stderr, "  bwt... done: %.2fs\n", timer.seconds());

        h_bwt = d_bwt;
        {
            // check whether the results match our expectations
            packed_stream_type h_packed_bwt_ref( nvbio::plain_view( h_bwt_ref ) );
            packed_stream_type h_packed_bwt( nvbio::plain_view( h_bwt ) );

            bool check = (primary_ref == primary);
            for (uint32 i = 0; i < N_symbols; ++i)
            {
                if (h_packed_bwt[i] != h_packed_bwt_ref[i])
                    check = false;
            }

            if (check == false)
            {
                dna_to_string(
                    packed_stream_type( nvbio::plain_view( h_bwt ) ),
                    N_symbols,
                    char_string );

                log_error(stderr, "mismatching results!\n" );
                log_error(stderr, "    primary : %u\n", primary );
                log_error(stderr, "    bwt     : %s\n", char_string );
                return 0u;
            }
        }
    }
    if (TEST_MASK & kGPU_BWT_GENOME)
    {
        // load a genome
        io::FMIndexDataRAM h_fmi;
        if (h_fmi.load( index_name, io::FMIndexData::GENOME | io::FMIndexData::FORWARD ) == false)
            return 0;

        // copy it to the gpu
        io::FMIndexDataCUDA d_fmi( h_fmi, io::FMIndexData::GENOME );

        typedef io::FMIndexData::stream_type                const_packed_stream_type;
        typedef io::FMIndexData::nonconst_stream_type             packed_stream_type;

        const uint32 N_symbols = d_fmi.genome_length();
        const uint32 N_words   = d_fmi.seq_words;

        fprintf(stderr, "  gpu bwt test\n");
        fprintf(stderr, "    %5.1f G symbols\n",  (1.0e-6f*float(N_symbols)));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        thrust::device_vector<uint32>  d_bwt( N_words+1 );

        const_packed_stream_type d_packed_string( d_fmi.genome_stream() );
              packed_stream_type d_packed_bwt( nvbio::plain_view( d_bwt ) );

        const uint32 primary_ref = cuda::find_primary( N_symbols, d_packed_string.begin() );
        fprintf(stderr, "    primary: %u\n", primary_ref);

        fprintf(stderr, "  bwt... started\n");

        Timer timer;
        timer.start();

        const uint32 primary = cuda::bwt(
            N_symbols,
            d_packed_string.begin(),
            d_packed_bwt.begin(),
            &params );

        timer.stop();

        fprintf(stderr, "  bwt... done: %.2fs\n", timer.seconds());

        bool check = primary == primary_ref;
        if (check == false)
        {
            log_error(stderr, "mismatching results!\n" );
            log_error(stderr, "    primary : %u\n", primary );
            return 0u;
        }
    }
    if (TEST_MASK & kGPU_BWT)
    {
        typedef PackedStream<uint32*,uint8,SYMBOL_SIZE,true,uint64>     packed_stream_type;
        typedef packed_stream_type::iterator                            packed_stream_iterator;

        const uint64 N_symbols  = 4llu*1024u*1024u*1024u - 1u;
        const uint64 N_words    = (N_symbols + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

        fprintf(stderr, "  gpu bwt test\n");
        fprintf(stderr, "    %5.1f G symbols\n",  (1.0e-9f*float(N_symbols)));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        thrust::host_vector<uint32>  h_string( N_words );

        LCG_random rand;
        for (uint64 i = 0; i < N_words; ++i)
            h_string[i] = rand.next();

        // insert some long common prefixes
        for (uint32 i = 50; i < 100; ++i)
            h_string[i] = 0;

        thrust::device_vector<uint32>  d_string( h_string );
        thrust::device_vector<uint32>  d_bwt( N_words );

        cudaDeviceSynchronize();

        packed_stream_type d_packed_string( nvbio::plain_view( d_string ) );
        packed_stream_type d_packed_bwt( nvbio::plain_view( d_bwt ) );

        fprintf(stderr, "  bwt... started\n");

        Timer timer;
        timer.start();

        cuda::bwt(
            N_symbols,
            d_packed_string.begin(),
            d_packed_bwt.begin(),
            &params );

        timer.stop();

        fprintf(stderr, "  bwt... done: %.2fs\n", timer.seconds());
    }
    if (TEST_MASK & kGPU_BWT_SET)
    {
        typedef uint32 word_type;
        typedef cuda::load_pointer<word_type,cuda::LOAD_DEFAULT> storage_type;

        typedef PackedStream<word_type*,uint8,SYMBOL_SIZE,true,uint64>      packed_stream_type;
        typedef packed_stream_type::iterator                                packed_stream_iterator;

        typedef PackedStream<storage_type,uint8,SYMBOL_SIZE,true,uint64>    mod_packed_stream_type;
        typedef mod_packed_stream_type::iterator                            mod_packed_stream_iterator;
        typedef ConcatenatedStringSet<mod_packed_stream_iterator,uint64*>   string_set;

        const uint32 N_strings  = gpu_bwt_size*1000*1000;
        const uint64 N_words    = (uint64(N_strings)*N + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

        fprintf(stderr, "  gpu set-bwt test\n");
        fprintf(stderr, "    %5.1f M strings\n",  (1.0e-6f*float(N_strings)));
        fprintf(stderr, "    %5.1f G suffixes\n", (1.0e-9f*float(uint64(N_strings)*uint64(N+1))));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        thrust::host_vector<uint32>  h_string( N_words );
        thrust::host_vector<uint64>  h_offsets( N_strings+1 );

        sufsort::make_test_string_set<SYMBOL_SIZE>(
            N_strings,
            N,
            h_string,
            h_offsets );

        thrust::device_vector<uint32>  d_string( h_string );
        thrust::device_vector<uint64>  d_offsets( h_offsets );

        cudaDeviceSynchronize();

        mod_packed_stream_type d_packed_string( storage_type( (word_type*)nvbio::plain_view( d_string ) ) );

        string_set d_string_set(
            N_strings,
            d_packed_string.begin(),
            nvbio::plain_view( d_offsets ) );

        fprintf(stderr, "  bwt... started\n");

        Timer timer;

        if (store_output)
        {
            thrust::device_vector<uint32> d_bwt( N_words );
            packed_stream_type            d_packed_bwt( (word_type*)nvbio::plain_view( d_bwt ) );

            DeviceBWTHandler<packed_stream_iterator> output_handler( d_packed_bwt.begin() );

            timer.start();

            cuda::bwt<SYMBOL_SIZE,true>(
                d_string_set,
                output_handler,
                &params );

            timer.stop();
        }
        else
        {
            DiscardBWTHandler output_handler;

            timer.start();

            cuda::bwt<SYMBOL_SIZE,true>(
                d_string_set,
                output_handler,
                &params );

            timer.stop();
        }

        fprintf(stderr, "  bwt... done: %.2fs\n", timer.seconds());
    }
    if (TEST_MASK & kCPU_BWT_SET)
    {
        typedef uint32 word_type;

        typedef PackedStream<word_type*,uint8,SYMBOL_SIZE,true,uint64>  packed_stream_type;
        typedef packed_stream_type::iterator                            packed_stream_iterator;
        typedef ConcatenatedStringSet<packed_stream_iterator,uint64*>   string_set;

        const uint32 N_strings  = cpu_bwt_size*1000*1000;
        const uint64 N_words    = (uint64(N_strings)*N + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

        fprintf(stderr, "  cpu set-bwt test\n");
        fprintf(stderr, "    %5.1f M strings\n",  (1.0e-6f*float(N_strings)));
        fprintf(stderr, "    %5.1f G suffixes\n", (1.0e-9f*float(uint64(N_strings)*uint64(N+1))));
        fprintf(stderr, "    %5.2f GB\n",         (float(N_words)*sizeof(uint32))/float(1024*1024*1024));

        thrust::host_vector<uint32>  h_string( N_words );
        thrust::host_vector<uint64>  h_offsets( N_strings+1 );

        sufsort::make_test_string_set<SYMBOL_SIZE>(
            N_strings,
            N,
            h_string,
            h_offsets );

        packed_stream_type h_packed_string( (word_type*)nvbio::plain_view( h_string ) );

        string_set h_string_set(
            N_strings,
            h_packed_string.begin(),
            nvbio::plain_view( h_offsets ) );

        fprintf(stderr, "  bwt... started\n");

        Timer timer;

        if (store_output)
        {
            thrust::host_vector<uint32>  h_bwt( N_words );
            packed_stream_type           h_packed_bwt( (word_type*)nvbio::plain_view( h_bwt ) );

            HostBWTHandler<packed_stream_iterator> output_handler( h_packed_bwt.begin() );

            timer.start();

            large_bwt<SYMBOL_SIZE,true>(
                h_string_set,
                output_handler,
                &params );

            timer.stop();
        }
        else
        {
            DiscardBWTHandler output_handler;

            timer.start();

            large_bwt<SYMBOL_SIZE,true>(
                h_string_set,
                output_handler,
                &params );

            timer.stop();
        }

        fprintf(stderr, "  bwt... done: %.2fs\n", timer.seconds());
    }
    fprintf(stderr, "nvbio/sufsort test... done\n");
    return 0;
}

} // namespace nvbio

int main(int argc, char* argv[])
{
    crcInit();

    int cuda_device = -1;
    int device_count;
    cudaGetDeviceCount(&device_count);
    log_verbose(stderr, "  cuda devices : %d\n", device_count);

    int arg = 1;
    if (argc > 1)
    {
        if (strcmp( argv[arg], "-device" ) == 0)
        {
            cuda_device = atoi(argv[++arg]);
            ++arg;
        }
    }

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

    // allocate some heap
    cudaDeviceSetLimit( cudaLimitMallocHeapSize, 128*1024*1024 );

    argc = argc >= arg ? argc-arg : 0;

    nvbio::sufsort_test( argc, argv+arg );

    cudaDeviceReset();
	return 0;
}
