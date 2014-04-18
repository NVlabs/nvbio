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

#include <nvbio/io/fmi.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/bnt.h>
#include <nvbio/basic/exceptions.h>
#include <nvbio/basic/dna.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fmindex/ssa.h>
#include <nvbio/fmindex/fmindex.h>
#include <crc/crc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <string>

#define FMI_ALIGNMENT 4u

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup FMIndexIO
///@{

namespace { // anonymous namespace

///@addtogroup FMIndexIODetail
///@{

template <typename T>
uint64 block_fread(T* dst, const uint64 n, FILE* file)
{
#if defined(WIN32)
    // use blocked reads on Windows, which seems to otherwise become less responsive while reading.
    const uint64 BATCH_SIZE = 16*1024*1024;
    for (uint64 batch_begin = 0; batch_begin < n; batch_begin += BATCH_SIZE)
    {
        const uint64 batch_end = nvbio::min( batch_begin + BATCH_SIZE, n );
        const uint64 batch_size = batch_end - batch_begin;

        const uint64 n_words = fread( dst + batch_begin, sizeof(T), batch_size, file );
        if (n_words != batch_size)
            return batch_begin + n_words;
    }
    return n;
#else
    return fread( dst, sizeof(T), n, file );
#endif
}

template <typename T>
void cuda_alloc(T*& dst, const T* src, const uint32 words, uint64& allocated)
{
    const uint32 words4 = 4u * ((words + 3u) / 4u);
    if (src)
    {
        cudaMalloc( &dst, sizeof(T) * words4 );
        if (dst == NULL)
        {
            size_t free, total;
            cudaMemGetInfo(&free, &total);
            throw nvbio::bad_alloc("FMIndexDataDevice: not enough device memory (allocation: %u MB, free: %u MB, total: %u MB)",
                sizeof(T)*words4/(1024*1024),
                free / (1024*1024), total / (1024*1024));
        }

        cudaMemcpy( dst, src, sizeof(T) * words, cudaMemcpyHostToDevice );

        allocated += words4 * sizeof(T);
    }
    else
        dst = NULL;
}

template <typename T>
void cuda_alloc(T*& dst, const uint32 words, uint64& allocated)
{
    const uint32 words4 = 4u * ((words + 3u) / 4u);

    cudaMalloc( &dst, sizeof(T) * words4 );
    if (dst == NULL)
    {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        throw nvbio::bad_alloc("FMIndexDataDevice: not enough device memory (allocation: %u MB, free: %u MB, total: %u MB)",
            sizeof(T)*words4/(1024*1024),
            free / (1024*1024), total / (1024*1024));
    }

    allocated += words4 * sizeof(T);
}

struct file_mismatch {};

struct VectorAllocator
{
    VectorAllocator(std::vector<uint32>& vec) : m_vec( vec ) {}

    uint32* alloc(const uint32 words)
    {
        m_vec.resize( words );
        return &m_vec[0];
    }

    std::vector<uint32>& m_vec;
};
struct MMapAllocator
{
    MMapAllocator(
        const char*       name,
        ServerMappedFile& mmap) : m_name( name ), m_mmap( mmap ) {}

    uint32* alloc(const uint32 words)
    {
        return (uint32*)m_mmap.init(
            m_name,
            words * sizeof(uint32),
            NULL );
    }

    const char*       m_name;
    ServerMappedFile& m_mmap;
};

template <typename Allocator>
uint32* load_genome(
    const char*     genome_prefix,
    Allocator&      allocator,
    uint32&         seq_length,
    uint32&         seq_words)
{
    std::string genome_wpac_string = std::string( genome_prefix ) + ".wpac";
    std::string genome_pac_string  = std::string( genome_prefix ) + ".pac";
    const char* wpac_file_name = genome_wpac_string.c_str();
    const char* pac_file_name  = genome_pac_string.c_str();

    typedef FMIndexData::nonconst_stream_type nonconst_stream_type;

    bool pac = false;

    FILE* genome_file = fopen( wpac_file_name, "rb" );
    if (genome_file == NULL)
    {
        genome_file = fopen( pac_file_name, "rb" );
        pac = true;
    }

    log_info(stderr, "reading (%s) genome... started\n", pac ? "pac" : "wpac");

    if (genome_file == NULL)
    {
        log_warning(stderr, "unable to open genome\n");
        return 0;
    }

    uint32* genome_stream = NULL;

    if (pac == false)
    {
        // read a .wpac file
        uint64 field;
        if (!fread( &field, sizeof(field), 1, genome_file ))
        {
            log_error(stderr, "error: failed reading genome\n");
            return 0;
        }
        seq_length = uint32(field);
        const uint32 unaligned_seq_words  = uint32( (seq_length+15)/16 );
        // make sure the genome length is a multiple of 4
        // this is required due to the interleaving of bwt and occ data in FMIndexDataDevice
        seq_words  = align<FMI_ALIGNMENT>( unaligned_seq_words );

        genome_stream = allocator.alloc( seq_words );
        // initialize the alignment slack
        for (uint32 i = unaligned_seq_words; i < seq_words; ++i)
            genome_stream[i] = 0u;

        const uint32 n_words = (uint32)block_fread( genome_stream, unaligned_seq_words, genome_file );
        if (n_words != unaligned_seq_words)
        {
            log_error(stderr, "error: failed reading genome\n");
            return 0;
        }
    }
    else
    {
        // read a .pac file
        fseek( genome_file, -1, SEEK_END );
        const uint32 packed_file_len = ftell( genome_file );
        uint8 last_byte_len;
        if (!fread( &last_byte_len, sizeof(unsigned char), 1, genome_file ))
        {
            log_error(stderr, "error: failed reading genome\n");
            return 0;
        }
        seq_length = (packed_file_len - 1u) * 4u + last_byte_len;

        fseek( genome_file, 0, SEEK_SET );

        const uint32 seq_bytes = uint32( (seq_length + 3u)/4u );

        std::vector<uint8> pac_vec( seq_bytes );
        uint8* pac_stream = &pac_vec[0];

        const uint64 n_bytes = block_fread( pac_stream, seq_bytes, genome_file );
        if (n_bytes != seq_bytes)
        {
            log_error(stderr, "error: failed reading genome\n");
            return 0;
        }

        // alloc the word-packed genome
        const uint32 unaligned_seq_words = uint32( (seq_length+15)/16 );
        // make sure the genome length is a multiple of 4
        // this is required due to the interleaving of bwt and occ data in FMIndexDataDevice
        seq_words = align<FMI_ALIGNMENT>( unaligned_seq_words );

        genome_stream = allocator.alloc( seq_words );
        // initialize the alignment slack
        for (uint32 i = unaligned_seq_words; i < seq_words; ++i)
            genome_stream[i] = 0u;

        // copy the pac stream into the genome
        typedef PackedStream<uint8*,uint8,2,true> pac_stream_type;
        pac_stream_type pac( pac_stream );

        nonconst_stream_type genome( genome_stream );
        for (uint32 i = 0; i < seq_length; ++i)
            genome[i] = pac[i];
    }
    fclose( genome_file );

    log_info(stderr, "reading (%s) genome... done\n", pac ? "pac" : "wpac");
    log_visible(stderr, "  genome length : %u bps (words: %u)\n", seq_length, seq_words);

    return genome_stream;
}

template <typename Allocator>
uint32* load_bwt(
    const char*     bwt_file_name,
    Allocator&      allocator,
    const uint32    seq_words,
    uint32&         primary)
{
    FILE* bwt_file = fopen( bwt_file_name, "rb" );
    if (bwt_file == NULL)
    {
        log_warning(stderr, "unable to open bwt \"%s\"\n", bwt_file_name);
        return 0;
    }
    uint32 field;
    if (!fread( &field, sizeof(field), 1, bwt_file ))
    {
        log_error(stderr, "error: failed reading bwt \"%s\"\n", bwt_file_name);
        return 0;
    }
    primary = uint32(field);

    // discard frequencies
    for (uint32 i = 0; i < 4; ++i)
    {
        if (!fread( &field, sizeof(field), 1, bwt_file ))
        {
            log_error(stderr, "error: failed reading bwt \"%s\"\n", bwt_file_name);
            return 0;
        }
    }

    uint32* bwt_stream = allocator.alloc( seq_words );

    const uint32 n_words = (uint32)block_fread( bwt_stream, seq_words, bwt_file );
    if (align<FMI_ALIGNMENT>(n_words) != seq_words)
    {
        log_error(stderr, "error: failed reading bwt \"%s\"\n", bwt_file_name);
        return 0;
    }
    // initialize the alignment slack
    for (uint32 i = n_words; i < seq_words; ++i)
        bwt_stream[i] = 0u;

    fclose( bwt_file );
    return bwt_stream;
}

template <typename Allocator>
uint32* load_sa(
    const char*     sa_file_name,
    Allocator&      allocator,
    const uint32    seq_length,
    const uint32    primary,
    const uint32    SA_INT)
{
    uint32* ssa = NULL;

    FILE* sa_file = fopen( sa_file_name, "rb" );
    if (sa_file != NULL)
    {
        log_info(stderr, "reading SSA... started\n");

        try
        {
            uint32 field;

            if (!fread( &field, sizeof(field), 1, sa_file ))
            {
                log_error(stderr, "error: failed reading SSA \"%s\"\n", sa_file_name);
                return 0;
            }
            if (field != primary)
            {
                log_error(stderr, "SA file mismatch \"%s\"\n", sa_file_name);
                throw file_mismatch();
            }

            for (uint32 i = 0; i < 4; ++i)
            {
                if (!fread( &field, sizeof(field), 1, sa_file ))
                {
                    log_error(stderr, "error: failed reading SSA \"%s\"\n", sa_file_name);
                    return 0;
                }
            }

            if (!fread( &field, sizeof(field), 1, sa_file ))
            {
                log_error(stderr, "error: failed reading SSA \"%s\"\n", sa_file_name);
                return 0;
            }
            if (field != SA_INT)
            {
                log_error(stderr, "unsupported SA interval (found %u, expected %u)\n", field, SA_INT);
                throw file_mismatch();
            }

            if(!fread( &field, sizeof(field), 1, sa_file ))
            {
                log_error(stderr, "error: failed reading SSA \"%s\"\n", sa_file_name);
                return 0;
            }
            if (field != seq_length)
            {
                log_error(stderr, "SA file mismatch \"%s\"\n", sa_file_name);
                throw file_mismatch();
            }

            const uint32 sa_size = (seq_length + SA_INT) / SA_INT;

            ssa = allocator.alloc( sa_size );
            ssa[0] = uint32(-1);
            if (!fread( &ssa[1], sizeof(uint32), sa_size-1, sa_file ))
            {
                log_error(stderr, "error: failed reading SSA \"%s\"\n", sa_file_name);
                return 0;
            }
        }
        catch (...)
        {
            // just skip the ssa file
        }
        fclose( sa_file );

        log_info(stderr, "reading SSA... done\n");
    }
    return ssa;
}

template <typename StringVector, typename AnnVector, typename AmbVector>
struct BNTLoader : public nvbio::BNTSeqLoader
{
    BNTLoader(
        StringVector& name_vec,
        StringVector& anno_vec,
        AnnVector&    ann_vec,
        AmbVector&    amb_vec) :
        m_name_vec( &name_vec ),
        m_anno_vec( &anno_vec ),
        m_ann_vec( &ann_vec ),
        m_amb_vec( &amb_vec ) {}

    void set_info(const nvbio::BNTInfo info)
    {
        m_info = info;
    }
    void read_ann(const nvbio::BNTAnnInfo& info, nvbio::BNTAnnData& data)
    {
        const uint32 name_offset = (uint32)m_name_vec->size();
        const uint32 anno_offset = (uint32)m_anno_vec->size();
        m_name_vec->resize( name_offset + info.name.length() + 1u );
        m_anno_vec->resize( anno_offset + info.anno.length() + 1u );

        strcpy( &m_name_vec->front() + name_offset, info.name.c_str() );
        strcpy( &m_anno_vec->front() + anno_offset, info.anno.c_str() );

        BNTAnn _ann;
        _ann.name_offset = name_offset;
        _ann.anno_offset = anno_offset;
        _ann.offset = data.offset;
        _ann.len    = data.len;
        _ann.n_ambs = data.n_ambs;
        _ann.gi     = data.gi;

        m_ann_vec->push_back( _ann );
    }
    void read_amb(const nvbio::BNTAmb& amb)
    {
        BNTAmb _amb;
        _amb.offset = amb.offset;
        _amb.len    = amb.len;
        _amb.amb    = amb.amb;

        m_amb_vec->push_back( _amb );
    }

    nvbio::BNTInfo  m_info;
    StringVector*   m_name_vec;
    StringVector*   m_anno_vec;
    AnnVector*      m_ann_vec;
    AmbVector*      m_amb_vec;
};

///@} // FMIndexIODetails

} // anonymous namespace

// constructor
//
FMIndexData::FMIndexData() :
    m_flags         ( 0 ),
    seq_length      ( 0 ),
    seq_words       ( 0 ),
    occ_words       ( 0 ),
    sa_words        ( 0 ),
    primary         ( 0 ),
    rprimary        ( 0 ),
    m_genome_stream ( NULL ),
    m_bwt_stream    ( NULL ),
    m_rbwt_stream   ( NULL ),
    m_occ           ( NULL ),
    m_rocc          ( NULL ),
    L2              ( NULL ),
    rL2             ( NULL ),
    count_table     ( NULL )
{
}

int FMIndexDataRAM::load(
    const char* genome_prefix,
    const uint32 flags)
{
    log_visible(stderr, "FMIndexData: loading... started\n");
    log_visible(stderr, "  genome : %s\n", genome_prefix);

    // bind pointers to static vectors
    m_flags     = flags;
     L2         = &m_L2[0];
    rL2         = &m_rL2[0];
    count_table = &m_count_table[0];

    seq_length = seq_words = 0;

    std::string genome_wpac_string = std::string( genome_prefix ) + ".wpac";
    std::string genome_pac_string  = std::string( genome_prefix ) + ".pac";
    std::string bwt_string    = std::string( genome_prefix ) + ".bwt";
    std::string rbwt_string   = std::string( genome_prefix ) + ".rbwt";
    std::string sa_string     = std::string( genome_prefix ) + ".sa";
    std::string rsa_string    = std::string( genome_prefix ) + ".rsa";
    //const char* wpac_file_name = genome_wpac_string.c_str();
    //const char* pac_file_name  = genome_pac_string.c_str();
    const char* bwt_file_name  = bwt_string.c_str();
    const char* rbwt_file_name = rbwt_string.c_str();
    const char* sa_file_name   = sa_string.c_str();
    const char* rsa_file_name  = rsa_string.c_str();

    // read genome
    //if (flags & GENOME) // currently needed to get the total length
    {
        VectorAllocator allocator( m_genome_stream_vec );
        m_genome_stream = load_genome(
            genome_prefix,
            allocator,
            seq_length,
            seq_words );

        if (0)
        {
            stream_type genome( m_genome_stream );
            const uint32 crc = crcCalc( genome.begin(), uint32(seq_length) );
            log_info(stderr, "  crc           : %u\n", crc);
        }
    }

    if (flags & FORWARD)
    {
        // read bwt
        log_info(stderr, "reading bwt... started\n");
        {
            VectorAllocator allocator( m_bwt_stream_vec );
            m_bwt_stream = load_bwt(
                bwt_file_name,
                allocator,
                seq_words,
                primary );

            if (m_bwt_stream == NULL)
                return 0;
        }
        log_info(stderr, "reading bwt... done\n");
    }

    if (flags & REVERSE)
    {
        log_info(stderr, "reading rbwt... started\n");
        {
            VectorAllocator allocator( m_rbwt_stream_vec );
            m_rbwt_stream = load_bwt(
                rbwt_file_name,
                allocator,
                seq_words,
                rprimary );

            if (m_rbwt_stream == NULL)
                return 0;
        }
        log_info(stderr, "reading rbwt... done\n");
    }

    if (flags & FORWARD) log_visible(stderr, "   primary : %u\n", uint32(primary));
    if (flags & REVERSE) log_visible(stderr, "  rprimary : %u\n", uint32(rprimary));

    const uint32 OCC_INT = FMIndexData::OCC_INT;
    const uint32 SA_INT  = FMIndexData::SA_INT;

    const uint32 has_genome = (flags & GENOME)  ? 1u : 0;
    const uint32 has_fw     = (flags & FORWARD) ? 1u : 0;
    const uint32 has_rev    = (flags & REVERSE) ? 1u : 0;
    const uint32 has_sa     = (flags & SA)      ? 1u : 0;

    const uint64 memory_footprint =
        (has_genome + has_fw + has_rev) * sizeof(uint32)*seq_words +
        (has_fw + has_rev)              * sizeof(uint32)*4*uint64(seq_length+OCC_INT-1)/OCC_INT +
        has_sa * (has_fw + has_rev)     * sizeof(uint32)*uint64(seq_length+SA_INT)/SA_INT;

    log_visible(stderr, "  memory   : %.1f MB\n", float(memory_footprint)/float(1024*1024));

    stream_type bwt( m_bwt_stream );
    stream_type rbwt( m_rbwt_stream );

    occ_words = ((seq_length+OCC_INT-1) / OCC_INT) * 4;

    if ((flags & FORWARD) ||
        (flags & REVERSE))
    {
        log_info(stderr, "building occurrence tables... started\n");

        uint32 cnt[ 4 ];
        uint32 rcnt[ 4 ];

        if (flags & FORWARD)
        {
            m_occ_vec.resize( occ_words, 0u );
            m_occ = &m_occ_vec[0];

            build_occurrence_table<OCC_INT>(
                bwt.begin(),
                bwt.begin() + seq_length,
                m_occ,
                cnt );
        }
        if (flags & REVERSE)
        {
            m_rocc_vec.resize( occ_words, 0u );
            m_rocc = &m_rocc_vec[0];

            build_occurrence_table<OCC_INT>(
                rbwt.begin(),
                rbwt.begin() + seq_length,
                m_rocc,
                rcnt );
        }

        // compute the L2 tables
        L2[0] = 0;
        for (uint32 c = 0; c < 4; ++c)
            L2[c+1] = L2[c] + cnt[c];

        rL2[0] = 0;
        for (uint32 c = 0; c < 4; ++c)
            rL2[c+1] = rL2[c] + rcnt[c];

        log_info(stderr, "building occurrence tables... done\n");
    }
    else
    {
        // zero out the L2 tables
        for (uint32 c = 0; c < 5; ++c)
            L2[c] = rL2[c] = 0;
    }

    // read ssa
    if (flags & SA)
    {
        if (flags & FORWARD)
        {
            VectorAllocator allocator( m_ssa_vec );
            ssa.m_ssa = load_sa(
                sa_file_name,
                allocator,
                seq_length,
                primary,
                SA_INT );
        }
        // read rssa
        if (flags & REVERSE)
        {
            VectorAllocator allocator( m_rssa_vec );
            rssa.m_ssa = load_sa(
                rsa_file_name,
                allocator,
                seq_length,
                rprimary,
                SA_INT );
        }
        sa_words = (seq_length + SA_INT) / SA_INT;
    }

    gen_bwt_count_table( count_table );

    // read the BNT sequence
    log_info(stderr, "reading BNT... started\n");
    {
        BNTLoader<
            std::vector<char>,
            std::vector<BNTAnn>,
            std::vector<BNTAmb> > loader(
            m_bnt_vec.names,
            m_bnt_vec.annos,
            m_bnt_vec.anns,
            m_bnt_vec.ambs );

        load_bns( &loader, genome_prefix );

        // store sequence info
        m_bnt_info.n_seqs     = loader.m_info.n_seqs;
        m_bnt_info.seed       = loader.m_info.seed;
        m_bnt_info.n_holes    = loader.m_info.n_holes;
        m_bnt_info.names_len  = (uint32)m_bnt_vec.names.size();
        m_bnt_info.annos_len  = (uint32)m_bnt_vec.annos.size();

        // setup pointers for each array
        m_bnt_data.names = &m_bnt_vec.names[0];
        m_bnt_data.annos = &m_bnt_vec.annos[0];
        m_bnt_data.anns  = &m_bnt_vec.anns[0];
        m_bnt_data.ambs  = &m_bnt_vec.ambs[0];
    }
    log_info(stderr, "reading BNT... done\n");

    log_visible(stderr, "FMIndexData: loading... done\n");
    return 1;
}

int FMIndexDataMMAPServer::load(const char* genome_prefix, const char* mapped_name)
{
    log_visible(stderr, "FMIndexData: loading... started\n");
    log_visible(stderr, "  genome : %s\n", genome_prefix);

    std::string genome_wpac_string = std::string( genome_prefix ) + ".wpac";
    std::string genome_pac_string  = std::string( genome_prefix ) + ".pac";
    std::string bwt_string    = std::string( genome_prefix ) + ".bwt";
    std::string rbwt_string   = std::string( genome_prefix ) + ".rbwt";
    std::string sa_string     = std::string( genome_prefix ) + ".sa";
    std::string rsa_string    = std::string( genome_prefix ) + ".rsa";
    const char* wpac_file_name = genome_wpac_string.c_str();
    //const char* pac_file_name  = genome_pac_string.c_str();
    const char* bwt_file_name  = bwt_string.c_str();
    const char* rbwt_file_name = rbwt_string.c_str();
    const char* sa_file_name   = sa_string.c_str();
    const char* rsa_file_name  = rsa_string.c_str();

    std::string infoName = std::string("nvbio.") + std::string( mapped_name ) + ".info";
    std::string pacName  = std::string("nvbio.") + std::string( mapped_name ) + ".pac";
    std::string bwtName  = std::string("nvbio.") + std::string( mapped_name ) + ".bwt";
    std::string rbwtName = std::string("nvbio.") + std::string( mapped_name ) + ".rbwt";
    std::string occName  = std::string("nvbio.") + std::string( mapped_name ) + ".occ";
    std::string roccName = std::string("nvbio.") + std::string( mapped_name ) + ".rocc";
    std::string saName   = std::string("nvbio.") + std::string( mapped_name ) + ".sa";
    std::string rsaName  = std::string("nvbio.") + std::string( mapped_name ) + ".rsa";
    std::string bntName  = std::string("nvbio.") + std::string( mapped_name ) + ".bnt";

    try
    {
        // read genome
        {
            MMapAllocator allocator( pacName.c_str(), m_pac_file );
            m_genome_stream = load_genome(
                genome_prefix,
                allocator,
                seq_length,
                seq_words );

            if (0)
            {
                stream_type genome( m_genome_stream );
                const uint32 crc = crcCalc( genome.begin(), uint32(seq_length) );
                log_info(stderr, "  crc           : %u\n", crc);
            }
        }

        // read bwt
        log_info(stderr, "reading bwt... started\n");
        {
            MMapAllocator allocator( bwtName.c_str(), m_bwt_file );
            m_bwt_stream = load_bwt(
                bwt_file_name,
                allocator,
                seq_words,
                primary );

            if (m_bwt_stream == NULL)
                return 0;
        }
        log_info(stderr, "reading bwt... done\n");

        log_info(stderr, "reading rbwt... started\n");
        {
            MMapAllocator allocator( rbwtName.c_str(), m_rbwt_file );
            m_rbwt_stream = load_bwt(
                rbwt_file_name,
                allocator,
                seq_words,
                rprimary );

            if (m_rbwt_stream == NULL)
                return 0;
        }
        log_info(stderr, "reading rbwt... done\n");

        log_visible(stderr, "   primary : %u\n", uint32(primary));
        log_visible(stderr, "  rprimary : %u\n", uint32(rprimary));

        const uint32 OCC_INT = FMIndexData::OCC_INT;
        const uint32 SA_INT  = FMIndexData::SA_INT;

        const uint64 memory_footprint =
            (wpac_file_name ? 3 : 2)*sizeof(uint32)*seq_words +
            2*sizeof(uint32)*4*uint64(seq_length+OCC_INT-1)/OCC_INT +
            2*sizeof(uint32)*uint64(seq_length+SA_INT)/SA_INT;

        log_visible(stderr, "  memory   : %.1f MB\n", float(memory_footprint)/float(1024*1024));

        stream_type bwt( m_bwt_stream );
        stream_type rbwt( m_rbwt_stream );

        occ_words = ((seq_length+OCC_INT-1) / OCC_INT) * 4;
        m_occ = (uint32*)m_occ_file.init(
            occName.c_str(),
            occ_words * sizeof(uint32),
            NULL );
        m_rocc = (uint32*)m_rocc_file.init(
            roccName.c_str(),
            occ_words * sizeof(uint32),
            NULL );

        uint32  cnt[ 4 ];
        uint32  rcnt[ 4 ];

        log_info(stderr, "building occurrence tables... started\n");
        build_occurrence_table<OCC_INT>(
            bwt.begin(),
            bwt.begin() + seq_length,
            m_occ,
            cnt );

        build_occurrence_table<OCC_INT>(
            rbwt.begin(),
            rbwt.begin() + seq_length,
            m_rocc,
            rcnt );
        log_info(stderr, "building occurrence tables... done\n");

        // read ssa
        {
            MMapAllocator allocator( saName.c_str(), m_sa_file );
            ssa.m_ssa = load_sa(
                sa_file_name,
                allocator,
                seq_length,
                primary,
                SA_INT );
        }
        // read rssa
        {
            MMapAllocator allocator( rsaName.c_str(), m_rsa_file );
            rssa.m_ssa = load_sa(
                rsa_file_name,
                allocator,
                seq_length,
                rprimary,
                SA_INT );
        }

        uint32 L2[5];
        L2[0] = 0;
        for (uint32 c = 0; c < 4; ++c)
            L2[c+1] = L2[c] + cnt[c];

        uint32 rL2[5];
        rL2[0] = 0;
        for (uint32 c = 0; c < 4; ++c)
            rL2[c+1] = rL2[c] + rcnt[c];

        sa_words = has_ssa() ? (seq_length + SA_INT) / SA_INT : 0u;

        // read the BNT sequence
        log_info(stderr, "reading BNT... started\n");
        {
            BNTSeqVec bnt;

            BNTLoader<
                std::vector<char>,
                std::vector<BNTAnn>,
                std::vector<BNTAmb> > loader(
                bnt.names,
                bnt.annos,
                bnt.anns,
                bnt.ambs );

            load_bns( &loader, genome_prefix );

            // store sequence info
            m_info.bnt.n_seqs     = loader.m_info.n_seqs;
            m_info.bnt.seed       = loader.m_info.seed;
            m_info.bnt.n_holes    = loader.m_info.n_holes;
            m_info.bnt.names_len  = (uint32)bnt.names.size();
            m_info.bnt.annos_len  = (uint32)bnt.annos.size();

            const size_t bnt_file_size =
                m_info.bnt.names_len +
                m_info.bnt.annos_len +
                m_info.bnt.n_seqs  * sizeof(BNTAnn) +
                m_info.bnt.n_holes * sizeof(BNTAmb);

            // allocate mapped memory
            uint8* mapped_storage = (uint8*)m_bnt_file.init(
                bntName.c_str(),
                bnt_file_size,
                NULL );

            // carve pointers for each array from the mapped memory arena
            BNTAnn* anns = (BNTAnn*)mapped_storage; mapped_storage += sizeof(BNTAnn) * m_info.bnt.n_seqs;
            BNTAmb* ambs = (BNTAmb*)mapped_storage; mapped_storage += sizeof(BNTAmb) * m_info.bnt.n_holes;
            char* names = (char*)mapped_storage; mapped_storage += m_info.bnt.names_len;
            char* annos = (char*)mapped_storage; mapped_storage += m_info.bnt.annos_len;

            // copy vectors into mapped memory arenas
            memcpy( anns, &bnt.anns[0], sizeof(BNTAnn) * m_info.bnt.n_seqs );
            memcpy( ambs, &bnt.ambs[0], sizeof(BNTAmb) * m_info.bnt.n_holes );
            memcpy( names, &bnt.names[0], m_info.bnt.names_len );
            memcpy( annos, &bnt.annos[0], m_info.bnt.annos_len );
        }
        log_info(stderr, "reading BNT... done\n");

        m_info.sequence_length = seq_length;
        m_info.sequence_words  = seq_words;
        m_info.occ_words       = occ_words;
        m_info.sa_words        = sa_words;
        m_info.primary         = primary;
        m_info.rprimary        = rprimary;
        for (uint32 i = 0; i < 5; ++i)
        {
            m_info.L2[i]  = L2[i];
            m_info.rL2[i] = rL2[i];
        }
        m_info_file.init(
            infoName.c_str(),
            sizeof(Info),
            &m_info );
    }
    catch (ServerMappedFile::mapping_error error)
    {
        log_error(stderr,"could not create file mapping object \"%s\" (error %d)\n",
            error.m_file_name,
            error.m_code );
    }
    catch (ServerMappedFile::view_error error)
    {
        log_error(stderr, "could not map view file \"%s\" (error %d)\n",
            error.m_file_name,
            error.m_code );
    }
    catch (...)
    {
    };

    log_visible(stderr, "FMIndexData: loading... done\n");
    return 1;
}

void init_ssa(
    const FMIndexData&       driver_data,
    FMIndexData::SSA_type&   ssa,
    FMIndexData::SSA_type&   rssa)
{
    typedef FMIndexData::rank_dict_type rank_dict_type;
    typedef FMIndexData::fm_index_type  fm_index_type;
    typedef FMIndexData::SSA_type       SSA_type;
    typedef FMIndexData::SSA_context    SSA_context;

    fm_index_type temp_fmi(
        driver_data.seq_length,
        driver_data.primary,
        driver_data.L2,
        rank_dict_type(
            driver_data.m_bwt_stream,
            driver_data.m_occ,
            driver_data.count_table ),
        SSA_context() );

    fm_index_type temp_rfmi(
        driver_data.seq_length,
        driver_data.rprimary,
        driver_data.rL2,
        rank_dict_type(
            driver_data.m_rbwt_stream,
            driver_data.m_rocc,
            driver_data.count_table ),
        SSA_context() );

    log_info(stderr, "building SSA... started\n");
    ssa = SSA_type( temp_fmi /*, SA_INT*/ );
    log_info(stderr, "building SSA... done\n");

    log_info(stderr, "building reverse SSA... started\n");
    rssa = SSA_type( temp_rfmi /*, SA_INT*/ );
    log_info(stderr, "building reverse SSA... done\n");
}


int FMIndexDataMMAP::load(
    const char* file_name)
{
    log_visible(stderr, "FMIndexData (MMAP) : loading... started\n");
    log_visible(stderr, "  genome : %s\n", file_name);

    std::string infoName = std::string("nvbio.") + std::string( file_name ) + ".info";
    std::string pacName  = std::string("nvbio.") + std::string( file_name ) + ".pac";
    std::string bwtName  = std::string("nvbio.") + std::string( file_name ) + ".bwt";
    std::string rbwtName = std::string("nvbio.") + std::string( file_name ) + ".rbwt";
    std::string occName  = std::string("nvbio.") + std::string( file_name ) + ".occ";
    std::string roccName = std::string("nvbio.") + std::string( file_name ) + ".rocc";
    std::string saName   = std::string("nvbio.") + std::string( file_name ) + ".sa";
    std::string rsaName  = std::string("nvbio.") + std::string( file_name ) + ".rsa";
    std::string bntName  = std::string("nvbio.") + std::string( file_name ) + ".bnt";

    // bind pointers to static vectors
     L2         = &m_L2[0];
    rL2         = &m_rL2[0];
    count_table = &m_count_table[0];

    try {
        const Info* info = (const Info*)m_info_file.init( infoName.c_str(), sizeof(Info) );

        const uint64 file_size     = info->sequence_words * sizeof(uint32);
        const uint64 occ_file_size = info->occ_words * sizeof(uint32);
        const uint64 sa_file_size  = info->sa_words * sizeof(uint32);

        m_genome_stream = (uint32*)m_genome_file.init( pacName.c_str(), file_size );
        m_bwt_stream    = (uint32*)m_bwt_file.init( bwtName.c_str(), file_size );
        m_rbwt_stream   = (uint32*)m_rbwt_file.init( rbwtName.c_str(), file_size );
        m_occ           = (uint32*)m_occ_file.init( occName.c_str(), occ_file_size );
        m_rocc          = (uint32*)m_rocc_file.init( roccName.c_str(), occ_file_size );
        if (info->sa_words)
        {
            ssa.m_ssa  = (uint32*)m_sa_file.init( saName.c_str(), sa_file_size );
            rssa.m_ssa = (uint32*)m_rsa_file.init( rsaName.c_str(), sa_file_size );
            sa_words   = (info->sequence_length + SA_INT) / SA_INT;
        }
        else
        {
            ssa.m_ssa  = NULL;
            rssa.m_ssa = NULL;
            sa_words   = 0u;
        }

        seq_length = info->sequence_length;
        seq_words  = info->sequence_words;
        occ_words  = info->occ_words;
        primary    = info->primary;
        rprimary   = info->rprimary;
        for (uint32 i = 0; i < 5; ++i)
        {
            L2[i]  = info->L2[i];
            rL2[i] = info->rL2[i];
        }

        m_bnt_info = info->bnt;

        const size_t bnt_file_size =
            info->bnt.names_len +
            info->bnt.annos_len +
            info->bnt.n_seqs  * sizeof(BNTAnn) +
            info->bnt.n_holes * sizeof(BNTAmb);

        // setup mapped memory client
        uint8* mapped_storage = (uint8*)m_bnt_file.init( bntName.c_str(), bnt_file_size );

        // get pointers to mapped memory arena
        m_bnt_data.anns = (BNTAnn*)mapped_storage; mapped_storage += sizeof(BNTAnn) * info->bnt.n_seqs;
        m_bnt_data.ambs = (BNTAmb*)mapped_storage; mapped_storage += sizeof(BNTAmb) * info->bnt.n_holes;
        m_bnt_data.names = (char*)mapped_storage; mapped_storage += info->bnt.names_len;
        m_bnt_data.annos = (char*)mapped_storage; mapped_storage += info->bnt.annos_len;
    }
    catch (MappedFile::mapping_error error)
    {
        log_error(stderr, "FMIndexDataMMAP: error mapping file \"%s\" (%d)!\n", error.m_file_name, error.m_code);
        return 0;
    }
    catch (MappedFile::view_error error)
    {
        log_error(stderr, "FMIndexDataMMAP: error viewing file \"%s\" (%d)!\n", error.m_file_name, error.m_code);
        return 0;
    }
    catch (...)
    {
        log_error(stderr, "FMIndexDataMMAP: error mapping file (unknown)!\n");
        return 0;
    }

    gen_bwt_count_table( count_table );

    log_visible(stderr, "FMIndexData (MMAP) : loading... done\n");
    return 1;
}

FMIndexDataDevice::FMIndexDataDevice(const FMIndexData& host_data, const uint32 flags) :
    m_allocated( 0u )
{
    seq_length = host_data.seq_length;
    seq_words  = host_data.seq_words;
    occ_words  = host_data.occ_words;
    sa_words   = host_data.sa_words;
    primary    = host_data.primary;
    rprimary   = host_data.rprimary;

    const uint32 sa_size = (seq_length + SA_INT) / SA_INT;

    m_genome_stream = NULL;
    m_bwt_stream    = NULL;
    m_rbwt_stream   = NULL;
    m_occ           = NULL;
    m_rocc          = NULL;
	ssa.m_ssa       = NULL;
	rssa.m_ssa      = NULL;

    if (flags & GENOME)
    {
        if (host_data.m_genome_stream == NULL)
            log_warning(stderr, "FMIndexDataDevice: requested genome is not available!\n");

        cuda_alloc( m_genome_stream, host_data.m_genome_stream, seq_words, m_allocated );
    }

    if (flags & FORWARD)
    {
        if (host_data.m_bwt_stream == NULL || host_data.m_occ == NULL)
            log_warning(stderr, "FMIndexDataDevice: requested forward BWT is not available!\n");

    #if defined(FUSED_BWT_OCC)
        thrust::host_vector<uint32> bwt_occ( seq_words + occ_words );

        if (occ_words < seq_words)  throw runtime_error("FMIndexDataDevice: occurrence table has %u words, BWT has %u!", occ_words, seq_words);
        if (occ_words % 4 != 0)     throw runtime_error("FMIndexDataDevice: occurrence table has %u words, not a multiple of 4!", occ_words);
        if (seq_words % 4 != 0)     throw runtime_error("FMIndexDataDevice: BWT has %u words, not a multiple of 4!", seq_words);

        for (uint32 w = 0; w < seq_words; w += 4)
        {
            bwt_occ[ w*2+0 ] = host_data.m_bwt_stream[ w+0 ];
            bwt_occ[ w*2+1 ] = host_data.m_bwt_stream[ w+1 ];
            bwt_occ[ w*2+2 ] = host_data.m_bwt_stream[ w+2 ];
            bwt_occ[ w*2+3 ] = host_data.m_bwt_stream[ w+3 ];
            bwt_occ[ w*2+4 ] = host_data.m_occ[ w+0 ];
            bwt_occ[ w*2+5 ] = host_data.m_occ[ w+1 ];
            bwt_occ[ w*2+6 ] = host_data.m_occ[ w+2 ];
            bwt_occ[ w*2+7 ] = host_data.m_occ[ w+3 ];
        }
        nvbio::cuda::thrust_copy_vector(m_bwt_occ, bwt_occ);
        m_allocated += sizeof(uint32)*(seq_words + occ_words);
    #else
        cuda_alloc( m_bwt_stream,    host_data.m_bwt_stream,    seq_words, m_allocated );
        cuda_alloc( m_occ,           host_data.m_occ,           occ_words, m_allocated );
    #endif
        if (flags & SA)
        {
            if (host_data.ssa.m_ssa == NULL)
                log_warning(stderr, "FMIndexDataDevice: requested forward SSA is not available!\n");

            cuda_alloc( const_cast<uint32*&>(ssa.m_ssa), host_data.ssa.m_ssa, sa_size, m_allocated );
        }
    }

    if (flags & REVERSE)
    {
        if (host_data.m_rbwt_stream == NULL || host_data.m_rocc == NULL)
            log_warning(stderr, "FMIndexDataDevice: requested reverse BWT is not available!\n");

    #if defined(FUSED_BWT_OCC)
        thrust::host_vector<uint32> bwt_occ( seq_words + occ_words );

        if (occ_words < seq_words)  throw runtime_error("FMIndexDataDevice: occurrence table has %u words, BWT has %u!", occ_words, seq_words);
        if (occ_words % 4 != 0)     throw runtime_error("FMIndexDataDevice: occurrence table has %u words, not a multiple of 4!", occ_words);
        if (seq_words % 4 != 0)     throw runtime_error("FMIndexDataDevice: BWT has %u words, not a multiple of 4!", seq_words);

        for (uint32 w = 0; w < seq_words; w += 4)
        {
            bwt_occ[ w*2+0 ] = host_data.m_rbwt_stream[ w+0 ];
            bwt_occ[ w*2+1 ] = host_data.m_rbwt_stream[ w+1 ];
            bwt_occ[ w*2+2 ] = host_data.m_rbwt_stream[ w+2 ];
            bwt_occ[ w*2+3 ] = host_data.m_rbwt_stream[ w+3 ];
            bwt_occ[ w*2+4 ] = host_data.m_rocc[ w+0 ];
            bwt_occ[ w*2+5 ] = host_data.m_rocc[ w+1 ];
            bwt_occ[ w*2+6 ] = host_data.m_rocc[ w+2 ];
            bwt_occ[ w*2+7 ] = host_data.m_rocc[ w+3 ];
        }
        nvbio::cuda::thrust_copy_vector(m_rbwt_occ, bwt_occ);
        m_allocated += sizeof(uint32)*(seq_words + occ_words);
    #else
        cuda_alloc( m_rbwt_stream,   host_data.m_rbwt_stream,   seq_words, m_allocated );
        cuda_alloc( m_rocc,          host_data.m_rocc,          occ_words, m_allocated );
    #endif
        if (flags & SA)
        {
            if (host_data.rssa.m_ssa == NULL)
                log_warning(stderr, "FMIndexDataDevice: requested reverse SSA is not available!\n");

            cuda_alloc( const_cast<uint32*&>(rssa.m_ssa), host_data.rssa.m_ssa, sa_size, m_allocated );
        }
    }

    cuda_alloc(  L2, host_data.L2,  5u, m_allocated );
    cuda_alloc( rL2, host_data.rL2, 5u, m_allocated );
    cuda_alloc( count_table, host_data.count_table, 256u, m_allocated );
    nvbio::cuda::check_error("FMIndexDataDevice");
}
FMIndexDataDevice::~FMIndexDataDevice()
{
    cudaFree( m_genome_stream );
    cudaFree( m_bwt_stream );
    cudaFree( m_rbwt_stream );
    cudaFree( m_occ );
    cudaFree( m_rocc );
    cudaFree( const_cast<uint32*>(ssa.m_ssa) );
    cudaFree( const_cast<uint32*>(rssa.m_ssa) );

    cudaFree( L2 );
    cudaFree( rL2 );
    cudaFree( count_table );
}

void init_ssa(
    const FMIndexDataDevice&              driver_data,
    FMIndexDataDevice::SSA_device_type&   ssa,
    FMIndexDataDevice::SSA_device_type&   rssa)
{
    typedef FMIndexData::rank_dict_type rank_dict_type;
    typedef FMIndexData::fm_index_type  fm_index_type;
    typedef FMIndexData::SSA_type       SSA_type;
    typedef FMIndexData::SSA_context    SSA_context;

    fm_index_type temp_fmi(
        driver_data.seq_length,
        driver_data.primary,
        driver_data.L2,
        rank_dict_type(
            driver_data.m_bwt_stream,
            driver_data.m_occ,
            driver_data.count_table ),
        SSA_context() );

    fm_index_type temp_rfmi(
        driver_data.seq_length,
        driver_data.rprimary,
        driver_data.rL2,
        rank_dict_type(
            driver_data.m_rbwt_stream,
            driver_data.m_rocc,
            driver_data.count_table ),
        SSA_context() );

    log_info(stderr, "building SSA... started\n");
    ssa.init( temp_fmi );
    log_info(stderr, "building SSA... done\n");

    log_info(stderr, "building reverse SSA... started\n");
    rssa.init( temp_rfmi );
    log_info(stderr, "building reverse SSA... done\n");
}

///@} // FMIndexIO
///@} // IO

} // namespace io
} // namespace nvbio
