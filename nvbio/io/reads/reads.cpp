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

#include <nvbio/io/reads/reads.h>
#include <nvbio/io/reads/reads_priv.h>
#include <nvbio/io/reads/reads_fastq.h>
#include <nvbio/io/reads/sam.h>
#include <nvbio/io/reads/bam.h>
#include <nvbio/basic/console.h>
#include <cuda_runtime.h>

#include <string.h>

namespace nvbio {
namespace io {

// factory method to open a read file, tries to detect file type based on file name
ReadDataStream *open_read_file(const char*           read_file_name,
                               const QualityEncoding qualities,
                               const uint32          max_reads,
                               const uint32          truncate_read_len,
                               const ReadEncoding    flags)
{
    // parse out file extension; look for .fastq.gz, .fastq suffixes
    uint32 len = uint32( strlen(read_file_name) );
    bool is_gzipped = false;

    // do we have a .gz suffix?
    if (len >= strlen(".gz"))
    {
        if (strcmp(&read_file_name[len - strlen(".gz")], ".gz") == 0)
        {
            is_gzipped = true;
            len = uint32(len - strlen(".gz"));
        }
    }

    // check for fastq suffix
    if (len >= strlen(".fastq"))
    {
        if (strncmp(&read_file_name[len - strlen(".fastq")], ".fastq", strlen(".fastq")) == 0)
        {
            return new ReadDataFile_FASTQ_gz(read_file_name,
                                             qualities,
                                             max_reads,
                                             truncate_read_len,
                                             flags);
        }
    }

    if (len >= strlen(".fq"))
    {
        if (strncmp(&read_file_name[len - strlen(".fq")], ".fq", strlen(".fq")) == 0)
        {
            return new ReadDataFile_FASTQ_gz(read_file_name,
                                             qualities,
                                             max_reads,
                                             truncate_read_len,
                                             flags);
        }
    }

    // check for sam suffix
    if (len >= strlen(".sam"))
    {
        if (strncmp(&read_file_name[len - strlen(".sam")], ".sam", strlen(".sam")) == 0)
        {
            ReadDataFile_SAM *ret;

            ret = new ReadDataFile_SAM(read_file_name,
                                       max_reads,
                                       truncate_read_len,
                                       flags);

            if (ret->init() == false)
            {
                delete ret;
                return NULL;
            }

            return ret;
        }
    }

    // check for bam suffix
    if (len >= strlen(".bam"))
    {
        if (strncmp(&read_file_name[len - strlen(".bam")], ".bam", strlen(".bam")) == 0)
        {
            ReadDataFile_BAM *ret;

            ret = new ReadDataFile_BAM(read_file_name,
                                       max_reads,
                                       truncate_read_len,
                                       flags);

            if (ret->init() == false)
            {
                delete ret;
                return NULL;
            }

            return ret;
        }
    }

    // we don't actually know what this is; guess fastq
    log_warning(stderr, "could not determine file type for %s; guessing %sfastq\n", read_file_name, is_gzipped ? "compressed " : "");
    return new ReadDataFile_FASTQ_gz(read_file_name,
                                     qualities,
                                     max_reads,
                                     truncate_read_len,
                                     flags);
}

namespace { // anonymous

// converts ASCII characters for amino-acids into
// a 5 letter alphabet for { A, C, G, T, N }.
inline unsigned char nst_nt4_encode(unsigned char c)
{
    static unsigned char nst_nt4_table[256] = {
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

    return nst_nt4_table[c];
}

// convert a quality value in one of the supported encodings to Phred
inline unsigned char convert_to_phred_quality(const QualityEncoding encoding, const uint8 q)
{
    // this table maps Solexa quality values to Phred scale
    static unsigned char s_solexa_to_phred[] = {
        0, 1, 1, 1, 1, 1, 1, 2, 2, 3,
        3, 4, 4, 5, 5, 6, 7, 8, 9, 10,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
        90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
        130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
        140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
        170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189,
        190, 191, 192, 193, 194, 195, 196, 197, 198, 199,
        200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
        220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
        230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
        250, 251, 252, 253, 254, 255
    };

    switch(encoding)
    {
    case Phred:
        return q;

    case Phred33:
        return q - 33;

    case Phred64:
        return q - 64;

    case Solexa:
        return s_solexa_to_phred[q];
    }

    // gcc is dumb
    return q;
}

// complement a base pair
inline unsigned char complement_bp(unsigned char bp)
{
    switch(bp)
    {
    case 'A':
        return 'T';

    case 'a':
        return 't';

    case 'T':
        return 'A';

    case 't':
        return 'a';

    case 'G':
        return 'C';

    case 'g':
        return 'c';

    case 'C':
        return 'G';

    case 'c':
        return 'g';

    default:
        return bp;
    }
}

} // anonymous namespace

ReadDataRAM::ReadDataRAM()
  : ReadData()
{
    // xxx: these values are magic, need to document!
    m_read_vec.reserve( 8*1024*1024 );
    m_qual_vec.reserve( 64*1024*1024 );

    m_read_index_vec.reserve( 16*1024 );
    m_read_index_vec.resize( 1u );
    m_read_index_vec[0] = 0u;

    m_name_index_vec.reserve( 16*1024 );
    m_name_index_vec.resize( 1u );
    m_name_index_vec[0] = 0u;
}

// signals that the batch is complete
void ReadDataRAM::end_batch(void)
{
    assert(m_read_stream_words == (m_read_stream_len + 7) / 8);

    m_avg_read_len = (uint32) ceilf(float(m_read_stream_len) / float(m_n_reads));

    // set the stream pointers
    m_read_stream = &m_read_vec[0];
    m_qual_stream = &m_qual_vec[0];
    m_read_index = &m_read_index_vec[0];

    m_name_stream = &m_name_vec[0];
    m_name_index = &m_name_index_vec[0];
}

// add a read to this batch
void ReadDataRAM::push_back(uint32 read_len,
                            const char *name,
                            const uint8* read,
                            const uint8* quality,
                            QualityEncoding q_encoding,
                            uint32 truncate_read_len,
                            const ReadEncoding conversion_flags)
{
    // truncate read
    // xxx: should we do this silently?
    read_len = nvbio::min(read_len, truncate_read_len);

    assert(read_len);

    // resize the reads & quality buffers
    {
        const uint32 bps_per_word = 32 / ReadData::READ_BITS;
        const uint32 words = (m_read_stream_len + read_len + bps_per_word - 1) / bps_per_word;
        m_read_vec.resize(words);
        m_qual_vec.resize(m_read_stream_len + read_len);
        m_read_stream_words = words;
    }

    // encode the read data
    ReadData::read_stream_type stream(&m_read_vec[0]);
    for(uint32 i = 0; i < read_len; i++)
    {
        char bp = read[i];

        if (conversion_flags & COMPLEMENT)
        {
            bp = complement_bp(bp);
        }

        // xxx: note that we're pushing in reverse order by default
        // this is to be consistent with reads_fastq.cpp
        if (conversion_flags & REVERSE)
        {
            stream[m_read_stream_len + read_len - i - 1] = nst_nt4_encode(bp);
            m_qual_vec[m_read_stream_len + read_len - i - 1] = convert_to_phred_quality(q_encoding, quality[i]);
        } else {
            stream[m_read_stream_len + i] = nst_nt4_encode(bp);
            m_qual_vec[m_read_stream_len + i] = convert_to_phred_quality(q_encoding, quality[i]);
        }
    }

    // update read and bp counts
    m_n_reads++;
    m_read_stream_len += read_len;
    m_read_index_vec.push_back(m_read_stream_len);

    m_min_read_len = nvbio::min(m_min_read_len, read_len);
    m_max_read_len = nvbio::max(m_max_read_len, read_len);

    // store the read name
    const uint32 name_len = uint32(strlen(name));
    const uint32 name_offset = m_name_stream_len;

    m_name_vec.resize(name_offset + name_len + 1);
    strcpy(&m_name_vec[name_offset], name);

    m_name_stream_len += name_len + 1;
    m_name_index_vec.push_back(m_name_stream_len);
}

// utility function to alloc and copy a vector in device memory
template <typename T>
static void cudaAllocAndCopyVector(T*& dst, const T* src, const uint32 words, uint64& allocated)
{
    const uint32 words4 = 4u * ((words + 3u) / 4u);
    if (src)
    {
        cudaMalloc( &dst, sizeof(T) * words4 );
        if (dst == NULL)
            throw std::bad_alloc(WINONLY("ReadDataCUDA: not enough device memory"));

        cudaMemcpy( dst, src, sizeof(T) * words, cudaMemcpyHostToDevice );

        allocated += words4 * sizeof(T);
    }
    else
        dst = NULL;
}

ReadDataCUDA::ReadDataCUDA(const ReadData& host_data, const uint32 flags)
  : ReadData(),
    m_allocated( 0 )
{
    m_name_stream_len   = 0;
    m_name_stream       = NULL;
    m_name_index        = NULL;
    m_qual_stream       = NULL;
    m_read_stream       = NULL;
    m_read_index        = NULL;

    m_n_reads           = host_data.m_n_reads;
    m_read_stream_len   = 0;
    m_read_stream_words = 0;

    m_min_read_len = host_data.m_min_read_len;
    m_max_read_len = host_data.m_max_read_len;
    m_avg_read_len = host_data.m_avg_read_len;

    if (flags & READS)
    {
        m_read_stream_len   = host_data.m_read_stream_len;
        m_read_stream_words = host_data.m_read_stream_words;

        cudaAllocAndCopyVector( m_read_stream, host_data.m_read_stream, m_read_stream_words, m_allocated );
        cudaAllocAndCopyVector( m_read_index,  host_data.m_read_index,  m_n_reads+1, m_allocated );
    }
    if (flags & QUALS)
        cudaAllocAndCopyVector( m_qual_stream, host_data.m_qual_stream, m_read_stream_len, m_allocated );
}

ReadDataCUDA::~ReadDataCUDA()
{
    if (m_read_stream)
        cudaFree( m_read_stream );

    if (m_read_index)
        cudaFree( m_read_index );

    if (m_qual_stream)
        cudaFree( m_qual_stream );
}

// grab the next batch of reads into a host memory buffer
ReadData *ReadDataFile::next(const uint32 batch_size)
{
    const uint32 to_load = std::min(m_max_reads - m_loaded, batch_size);
    uint32 m;

    if (!is_ok() || to_load == 0)
        return NULL;

    ReadDataRAM *reads = new ReadDataRAM();

    m = 0;
    while (m < to_load)
    {
        // load 100 at a time if possible
        const uint32 q = std::min(to_load - m, uint32(100));
        int n;

        n = nextChunk(reads, q);
        assert(n <= (int) q);
        if (n == 0)
        {
            break;
        }

        m += n;
        assert(m <= to_load);
    }

    if (m == 0)
        return NULL;

    m_loaded += m;

    reads->end_batch();

    return reads;
}

} // namespace io
} // namespace nvbio
