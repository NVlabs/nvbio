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

#pragma once

#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_priv.h>
#include <nvbio/io/output_stream.h>
#include <nvbio/basic/console.h>

#include <zlib/zlib.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///@addtogroup SequenceIODetail
///@{

// SequenceDataFile from a FASTQ file
// contains the code to parse FASTQ files and dump the results into a SequenceDataRAM object
// file access is done via derived classes
struct SequenceDataFile_TXT : public SequenceDataFile
{
protected:
    SequenceDataFile_TXT(
        const char*     read_file_name,
        const Options&  options,
        const uint32    buffer_size = 64536u)
      : SequenceDataFile( options ),
        m_file_name(read_file_name),
        m_buffer(buffer_size),
        m_buffer_size(buffer_size),
        m_buffer_pos(buffer_size),
        m_line(0)
    {};

    // get next read chunk from file and parse it (up to max reads)
    // this can cause m_file_state to change
    virtual int nextChunk(struct SequenceDataEncoder* output, uint32 max_reads, uint32 max_bps);

    // fill m_buffer with data from the file, return the new file state
    // this should only report EOF when no more bytes could be read
    // derived classes should override this method to return actual file data
    virtual FileState fillBuffer(void) = 0;

    virtual bool gets(char* buffer, int len) = 0;

private:
    // get next character from file
    uint8 get();

protected:
    // file name we're reading from
    const char *            m_file_name;

    // buffers input from the fastq file
    std::vector<char>       m_buffer;
    uint32                  m_buffer_size;
    uint32                  m_buffer_pos;

    // counter for which line we're at
    uint32                  m_line;

    // error reporting from the parser: stores the character that generated an error
    uint8                   m_error_char;

    // temp buffers for data coming in from the FASTQ file: read name, base pairs and qualities
    std::vector<char>  m_name;
    std::vector<uint8> m_read_bp;
    std::vector<uint8> m_read_q;
};

// loader for gzipped files
// this also works for plain uncompressed files, as zlib does that transparently
struct SequenceDataFile_TXT_gz : public SequenceDataFile_TXT
{
    SequenceDataFile_TXT_gz(
        const char*     read_file_name,
        const Options&  options,
        const uint32    buffer_size = 64536u);

    ~SequenceDataFile_TXT_gz();

    /// fill the buffer
    ///
    virtual FileState fillBuffer(void);

    virtual bool gets(char* buffer, int len) { return gzgets( m_file, buffer, len ) != NULL; };

    /// rewind the file
    ///
    virtual bool rewind();

private:
    gzFile m_file;
};

/// SequenceDataFile from a FASTQ file
/// contains the code to parse FASTQ files and dump the results into a SequenceDataRAM object
/// file access is done via derived classes
///
struct SequenceDataOutputFile_TXT : SequenceDataOutputStream
{
    /// constructor
    ///
    SequenceDataOutputFile_TXT(
        const char* file_name,
        const char* compressor,
        const char* options);

    /// next batch
    ///
    void next(const SequenceDataHost& sequence_data);

    /// return whether the stream is ok
    ///
    bool is_ok();

private:
    // file name we're reading from
    const char*   m_file_name;
    OutputStream* m_file;
};

///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

inline uint8 SequenceDataFile_TXT::get(void)
{
    if (m_buffer_pos >= m_buffer_size)
    {
        // grab more data from the underlying file
        m_file_state = fillBuffer();
        m_buffer_pos = 0;

        // if we failed to read more data, return \0
        if (m_file_state != FILE_OK)
            return 0;
    }

    return m_buffer[m_buffer_pos++];
}

} // namespace io
} // namespace nvbio
