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

#include <nvbio/io/sequence/sequence_fastq.h>
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

int SequenceDataFile_FASTQ_parser::nextChunk(SequenceDataEncoder *output, uint32 max_reads, uint32 max_bps)
{
    uint32 n_reads = 0;
    uint32 n_bps   = 0;
    uint8  marker;

    const uint32 read_mult =
        ((m_options.flags & FORWARD)            ? 1u : 0u) +
        ((m_options.flags & REVERSE)            ? 1u : 0u) +
        ((m_options.flags & FORWARD_COMPLEMENT) ? 1u : 0u) +
        ((m_options.flags & REVERSE_COMPLEMENT) ? 1u : 0u);

    while (n_reads + read_mult                         <= max_reads &&
           n_bps   + read_mult*SequenceDataFile::LONG_READ <= max_bps)
    {
        // consume spaces & newlines
        do {
            marker = get();

            // count lines
            if (marker == '\n')
                m_line++;
        }
        while (marker == '\n' || marker == ' ');

        // check for EOF or read errors
        if (m_file_state != FILE_OK)
            break;

        // if the newlines didn't end in a read marker,
        // issue a parsing error...
        if (marker != '@')
        {
            m_file_state = FILE_PARSE_ERROR;
            m_error_char = marker;
            return uint32(-1);
        }

        // read all the line
        uint32 len = 0;
        for (uint8 c = get(); c != '\n' && c != 0; c = get())
        {
            m_name[ len++ ] = c;

            // expand on demand
            if (m_name.size() <= len)
                m_name.resize( len * 2u );
        }

        m_name[ len++ ] = '\0';

        // check for errors
        if (m_file_state != FILE_OK)
        {
            log_error(stderr, "incomplete read!\n");

            m_error_char = 0;
            return uint32(-1);
        }

        m_line++;

        // start reading the bp read
        len = 0;
        for (uint8 c = get(); c != '+' && c != 0; c = get())
        {
            // if (isgraph(c))
            if (c >= 0x21 && c <= 0x7E)
                m_read_bp[ len++ ] = c;
            else if (c == '\n')
                m_line++;

            // expand on demand
            if (m_read_bp.size() <= len)
            {
                m_read_bp.resize( len * 2u );
                m_read_q.resize(  len * 2u );
            }
        }

        // check for errors
        if (m_file_state != FILE_OK)
        {
            log_error(stderr, "incomplete read!\n");

            m_error_char = 0;
            return uint32(-1);
        }

        // read all the line
        for(uint8 c = get(); c != '\n' && c != 0; c = get()) {}

        // check for errors
        if (m_file_state != FILE_OK)
        {
            log_error(stderr, "incomplete read!\n");

            m_error_char = 0;
            return uint32(-1);
        }

        m_line++;

        // start reading the quality read
        len = 0;
        for (uint8 c = get(); c != '\n' && c != 0; c = get())
            m_read_q[ len++ ] = c;

        // check for errors
        if (m_file_state != FILE_OK)
        {
            log_error(stderr, "incomplete read!\n");

            m_error_char = 0;
            return uint32(-1);
        }

        m_line++;

        if (m_options.flags & FORWARD)
        {
            output->push_back( len,
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_options.qualities,
                              m_options.max_sequence_len,
                              m_options.trim3,
                              m_options.trim5,
                              SequenceDataEncoder::NO_OP );
        }
        if (m_options.flags & REVERSE)
        {
            output->push_back( len,
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_options.qualities,
                              m_options.max_sequence_len,
                              m_options.trim3,
                              m_options.trim5,
                              SequenceDataEncoder::REVERSE_OP );
        }
        if (m_options.flags & FORWARD_COMPLEMENT)
        {
            output->push_back( len,
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_options.qualities,
                              m_options.max_sequence_len,
                              m_options.trim3,
                              m_options.trim5,
                              SequenceDataEncoder::COMPLEMENT_OP );
        }
        if (m_options.flags & REVERSE_COMPLEMENT)
        {
            output->push_back( len,
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_options.qualities,
                              m_options.max_sequence_len,
                              m_options.trim3,
                              m_options.trim5,
                              SequenceDataEncoder::REVERSE_COMPLEMENT_OP );
        }

        n_bps   += read_mult * len;
        n_reads += read_mult;
    }
    return n_reads;
}

SequenceDataFile_FASTQ_gz::SequenceDataFile_FASTQ_gz(
    const char*                         read_file_name,
    const SequenceDataFile::Options&    options)
    : SequenceDataFile_FASTQ_parser(read_file_name, options)
{
    m_file = gzopen(read_file_name, "r");
    if (!m_file) {
        m_file_state = FILE_OPEN_FAILED;
    } else {
        m_file_state = FILE_OK;
    }

    gzbuffer(m_file, m_buffer_size);
}

SequenceDataFile_FASTQ_gz::~SequenceDataFile_FASTQ_gz()
{
    gzclose( m_file );
}

//static float time = 0.0f;

SequenceDataFile_FASTQ_parser::FileState SequenceDataFile_FASTQ_gz::fillBuffer(void)
{
    m_buffer_size = gzread(m_file, &m_buffer[0], (uint32)m_buffer.size());

    if (m_buffer_size <= 0)
    {
        // check for EOF separately; zlib will not always return Z_STREAM_END at EOF below
        if (gzeof(m_file))
        {
            return FILE_EOF;
        } else {
            // ask zlib what happened and inform the user
            int err;
            const char *msg;

            msg = gzerror(m_file, &err);
            // we're making the assumption that we never see Z_STREAM_END here
            assert(err != Z_STREAM_END);

            log_error(stderr, "error processing FASTQ file: zlib error %d (%s)\n", err, msg);
            return FILE_STREAM_ERROR;
        }
    }
    return FILE_OK;
}

// rewind
//
bool SequenceDataFile_FASTQ_gz::rewind()
{
    if (m_file == NULL)
        return false;

    gzrewind( m_file );

    m_file_state = FILE_OK;

    m_buffer_size = 0;
    m_buffer_pos  = 0;
    m_line        = 0;
    return true;
}

// constructor
//
SequenceDataOutputFile_FASTQ::SequenceDataOutputFile_FASTQ(
    const char* file_name,
    const char* compressor,
    const char* options)
  : m_file_name(file_name)
{
    m_file = open_output_file( file_name, compressor, options );
}

namespace {

template <Alphabet ALPHABET>
void write(
   OutputStream*                            output_file,
   const io::SequenceDataAccess<ALPHABET>&  sequence_data)
{
    typedef typename io::SequenceDataAccess<ALPHABET>::sequence_string  sequence_string;
    typedef typename io::SequenceDataAccess<ALPHABET>::qual_string      qual_string;
    typedef typename io::SequenceDataAccess<ALPHABET>::name_string      name_string;

    std::vector<char> buffer( 1024*1024 );

    for (uint32 i = 0; i < sequence_data.size(); ++i)
    {
        const sequence_string read = sequence_data.get_read( i );
        const qual_string     qual = sequence_data.get_quals( i );
        const name_string     name = sequence_data.get_name( i );

        uint32 buffer_len = 0;

        buffer[ buffer_len++ ] = '@';

        for (uint32 j = 0; j < name.size(); ++j)
            buffer[ buffer_len++ ] = name[j];

        // setup the ASCII read
        buffer[ buffer_len++ ] = '\n';

        to_string<ALPHABET>( read.begin(), read.size(), &buffer[0] + buffer_len );

        buffer_len += read.size();

        buffer[ buffer_len++ ] = '\n';
        buffer[ buffer_len++ ] = '+';
        buffer[ buffer_len++ ] = '\n';

        // copy the qualities to a null-terminated ASCII string
        for (uint32 j = 0; j < read.size(); ++j)
            buffer[ buffer_len++ ] = char( qual[j] );

        buffer[ buffer_len++ ] = '\n';

        if (output_file->write( buffer_len, &buffer[0] ) == 0 );
            throw runtime_error( "failed writing FASTQ output file" );
    }
}

} // anonymous namespace

// next batch
//
void SequenceDataOutputFile_FASTQ::next(const SequenceDataHost& sequence_data)
{
    if (sequence_data.alphabet() == DNA)
        write( m_file, io::SequenceDataAccess<DNA>( sequence_data ) );
    else if (sequence_data.alphabet() == DNA_N)
        write( m_file, io::SequenceDataAccess<DNA_N>( sequence_data ) );
    else if (sequence_data.alphabet() == PROTEIN)
        write( m_file, io::SequenceDataAccess<PROTEIN>( sequence_data ) );
    else if (sequence_data.alphabet() == RNA)
        write( m_file, io::SequenceDataAccess<RNA>( sequence_data ) );
    else if (sequence_data.alphabet() == RNA_N)
        write( m_file, io::SequenceDataAccess<RNA_N>( sequence_data ) );
    else if (sequence_data.alphabet() == ASCII)
        write( m_file, io::SequenceDataAccess<ASCII>( sequence_data ) );
}

// return whether the stream is ok
//
bool SequenceDataOutputFile_FASTQ::is_ok() { return m_file->is_valid(); }


///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
