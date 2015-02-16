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

#include <nvbio/io/sequence/sequence_fasta.h>
#include <nvbio/io/sequence/sequence_encoder.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/timer.h>

#include <string.h>
#include <ctype.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///@addtogroup SequenceIODetail
///@{

namespace { // anonymous namespace

struct FASTAHandler
{
    FASTAHandler(
        SequenceDataEncoder*             output,
        const SequenceDataFile::Options& options) :
        m_output( output ),
        m_options( options ) {}

    void push_back(const char* id, const uint32 read_len, const uint8* bp)
    {
        if (m_quals.size() < size_t( read_len ))
            m_quals.resize( read_len, 50u );

        if (m_options.flags & FORWARD)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                m_options.qualities,
                m_options.max_sequence_len,
                m_options.trim3,
                m_options.trim5,
                SequenceDataEncoder::NO_OP );
        }
        if (m_options.flags & REVERSE)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                m_options.qualities,
                m_options.max_sequence_len,
                m_options.trim3,
                m_options.trim5,
                SequenceDataEncoder::REVERSE_OP );
        }
        if (m_options.flags & FORWARD_COMPLEMENT)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                m_options.qualities,
                m_options.max_sequence_len,
                m_options.trim3,
                m_options.trim5,
                SequenceDataEncoder::COMPLEMENT_OP );
        }
        if (m_options.flags & REVERSE_COMPLEMENT)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                m_options.qualities,
                m_options.max_sequence_len,
                m_options.trim3,
                m_options.trim5,
                SequenceDataEncoder::REVERSE_COMPLEMENT_OP );
        }
    }

    SequenceDataEncoder*                m_output;
    const SequenceDataFile::Options     m_options;
    std::vector<uint8>                  m_quals;
};

} // anonymous namespace

// constructor
//
SequenceDataFile_FASTA_gz::SequenceDataFile_FASTA_gz(
    const char*                         read_file_name,
    const SequenceDataFile::Options&    options) :
    SequenceDataFile( options ),
    m_fasta_reader( read_file_name )
{
	if (!m_fasta_reader.valid()) {
		m_file_state = FILE_OPEN_FAILED;
	} else {
		m_file_state = FILE_OK;
	}
}

// rewind
//
bool SequenceDataFile_FASTA_gz::rewind()
{
    m_fasta_reader.rewind();
    m_file_state = FILE_OK;
    return true;
}

// get a chunk of reads
//
int SequenceDataFile_FASTA_gz::nextChunk(SequenceDataEncoder *output, uint32 max_reads, uint32 max_bps)
{
    const uint32 read_mult =
        ((m_options.flags & FORWARD)            ? 1u : 0u) +
        ((m_options.flags & REVERSE)            ? 1u : 0u) +
        ((m_options.flags & FORWARD_COMPLEMENT) ? 1u : 0u) +
        ((m_options.flags & REVERSE_COMPLEMENT) ? 1u : 0u);

    // build a writer
    FASTAHandler writer( output, m_options );

    return m_fasta_reader.read( max_reads / read_mult, writer );
}

///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
