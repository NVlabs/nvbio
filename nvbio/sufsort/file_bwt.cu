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

#include <nvbio/sufsort/file_bwt.h>
#include <nvbio/sufsort/file_bwt_bgz.h>
#include <nvbio/sufsort/sufsort_priv.h>
#include <zlib/zlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace nvbio {

namespace { // anonymous namespace

/// convert a DNA+N+$ symbol to its ASCII character
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char dna6_to_char(const uint8 c)
{
    return c == 0u   ? 'A' :
           c == 1u   ? 'C' :
           c == 2u   ? 'G' :
           c == 3u   ? 'T' :
           c == 255u ? '$' :
                       'N';
}

/// convert a DNA+N+$ string to an ASCII string
///
template <typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void dna6_to_string(
    const SymbolIterator    begin,
    const uint32            n,
    char*                   string)
{
    for (uint32 i = 0; i < n; ++i)
        string[i] = dna6_to_char( begin[i] );

    string[n] = '\0';
}

// utility function to convert an unsigned int to a base-10 string representation
template <typename T> uint32 itoa(char *buf, T in)
{
    uint32 len = 0;

    // convert to base10
    do
    {
        buf[len] = "0123456789"[in % 10];
        in /= 10;
        len++;
    } while(in);

    // reverse
    for(uint32 c = 0; c < len / 2; c++)
    {
        char tmp;
        tmp = buf[c];
        buf[c] = buf[len - c - 1];
        buf[len - c - 1] = tmp;
    }

    // terminate
    buf[len] = 0;
    return len;
}

} // anonymous namespace

#define GPU_RANKING

/// Map each dollar's rank to its sequence index
///
struct DollarRankMap
{
    typedef std::pair<uint64,uint32> entry_type;

    /// constructor
    ///
    DollarRankMap() :
        offset(0),
        n_dollars(0) {}

    /// process a batch of BWT symbols
    ///
    uint32 extract(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        if (h_suffixes != NULL &&   // these are NULL for the empty suffixes
            d_suffixes != NULL)
        {
        #if defined(GPU_RANKING)
            priv::alloc_storage( found_dollars,    n_suffixes );
            priv::alloc_storage( d_dollar_ranks,   n_suffixes );
            priv::alloc_storage( d_dollars,        n_suffixes );
            priv::alloc_storage( h_dollar_ranks,   n_suffixes );
            priv::alloc_storage( h_dollars,        n_suffixes );

            uint32 n_found_dollars = 0;

            if (d_indices != NULL)
            {
                priv::alloc_storage( d_dollar_indices, n_suffixes );

                // find the dollar signs
                n_found_dollars = cuda::copy_flagged(
                    n_suffixes,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            thrust::make_counting_iterator<uint32>(0),
                            thrust::device_ptr<const uint32>( d_indices ) ) ),
                    thrust::make_transform_iterator( thrust::device_ptr<const uint8>( d_bwt ), equal_to_functor<uint8>(255u) ),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            d_dollar_ranks.begin(),
                            d_dollar_indices.begin() ) ),
                    d_temp_storage );

                // gather their indices
                thrust::gather(
                    d_dollar_indices.begin(),
                    d_dollar_indices.begin() + n_found_dollars,
                    thrust::make_transform_iterator( thrust::device_ptr<const uint2>( d_suffixes ), priv::suffix_component_functor<priv::STRING_ID>() ),
                    d_dollars.begin() );
            }
            else
            {
                // find the dollar signs
                n_found_dollars = cuda::copy_flagged(
                    n_suffixes,
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            thrust::make_counting_iterator<uint32>(0),
                            thrust::make_transform_iterator( thrust::device_ptr<const uint2>( d_suffixes ), priv::suffix_component_functor<priv::STRING_ID>() ) ) ),
                    thrust::make_transform_iterator( thrust::device_ptr<const uint8>( d_bwt ), equal_to_functor<uint8>(255u) ),
                    thrust::make_zip_iterator(
                        thrust::make_tuple(
                            d_dollar_ranks.begin(),
                            d_dollars.begin() ) ),
                    d_temp_storage );
            }

            // and copy them back to the host
            thrust::copy(
                d_dollar_ranks.begin(),
                d_dollar_ranks.begin() + n_found_dollars,
                h_dollar_ranks.begin() );

            // and copy them back to the host
            thrust::copy(
                d_dollars.begin(),
                d_dollars.begin() + n_found_dollars,
                h_dollars.begin() );

            #pragma omp parallel for
            for (int i = 0; i < int( n_found_dollars ); ++i)
            {
                found_dollars[i] = std::make_pair(
                    uint64( offset + h_dollar_ranks[i] ),
                    h_dollars[i] );
            }

            n_dollars += n_found_dollars;
            return n_found_dollars;
        #else
            priv::alloc_storage( found_dollars, n_suffixes );
            priv::alloc_storage( h_indices,     n_suffixes );

            const priv::suffix_component_functor<priv::STRING_ID> suffix_string;

            uint32 n_found_dollars = 0;

            if (d_indices != NULL)
            {
                // copy the indices back to the host
                thrust::copy(
                    thrust::device_ptr<const uint32>( d_indices ),
                    thrust::device_ptr<const uint32>( d_indices ) + n_suffixes,
                    h_indices.begin() );

                // loop through every symbol and keep track of the dollars
                for (uint32 i = 0; i < n_suffixes; ++i)
                {
                    if (h_bwt[i] == 255u)
                    {
                        h_dollars[ n_found_dollars++ ] = std::make_pair(
                            uint64( offset + i ),
                            suffix_string( h_suffixes[ h_indices[i] ] ) );
                    }
                }
            }
            else
            {
                // loop through every symbol and keep track of the dollars
                for (uint32 i = 0; i < n_suffixes; ++i)
                {
                    if (h_bwt[i] == 255u)
                    {
                        found_dollars[ n_found_dollars++ ] = std::make_pair(
                            uint64( offset + i ),
                            suffix_string( h_suffixes[i] ) );
                    }
                }
            }

            n_dollars += n_found_dollars;
            return n_found_dollars;
        #endif
        }

        offset += n_suffixes;
        return 0;
    }

    uint64                          offset;
    uint32                          n_dollars;
    std::vector<entry_type>         found_dollars;

  #if defined(GPU_RANKING)
    thrust::device_vector<uint32>   d_dollar_ranks;
    thrust::device_vector<uint32>   d_dollar_indices;
    thrust::device_vector<uint32>   d_dollars;
    thrust::host_vector<uint32>     h_dollar_ranks;
    thrust::host_vector<uint32>     h_dollars;
    thrust::device_vector<uint8>    d_temp_storage;
  #else
    thrust::host_vector<uint32>     h_indices;
  #endif
};

/// A class to output the BWT to a packed host string
///
template <typename BWTWriter, uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename word_type>
struct FileBWTHandler : public BaseBWTHandler, public BWTWriter
{
    static const uint32 WORD_SIZE = uint32( 8u * sizeof(word_type) );
    static const uint32 SYMBOLS_PER_WORD = WORD_SIZE / SYMBOL_SIZE;

    /// constructor
    ///
    FileBWTHandler() : offset(0) {}

    /// destructor
    ///
    virtual ~FileBWTHandler() {}

    /// write header
    ///
    void write_header()
    {
        const char* magic = "PRIB";         // PRImary-Binary
        BWTWriter::index_write( 4, magic );
    }

    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        const uint32 n_words = util::round_i( n_suffixes, SYMBOLS_PER_WORD );

        // expand our cache if needed
        if (cache.size() < n_words+2 ) // 2 more guardband words to avoid out-of-bounds accesses
            cache.resize( n_words+2 );

        const uint32 word_offset = offset & (SYMBOLS_PER_WORD-1);
              uint32 word_rem    = 0;
              uint32 cache_idx   = 0;

        if (word_offset)
        {
            // compute how many symbols we still need to encode to fill the current word
            word_rem = SYMBOLS_PER_WORD - word_offset;

            // fetch the word in question
            word_type word = cache_word;

            for (uint32 i = 0; i < word_rem; ++i)
            {
                const uint32       bit_idx = (word_offset + i) * SYMBOL_SIZE;
                const uint32 symbol_offset = BIG_ENDIAN ? (WORD_SIZE - SYMBOL_SIZE - bit_idx) : bit_idx;
                const word_type     symbol = word_type(h_bwt[i]) << symbol_offset;

                // set bits
                word |= symbol;
            }

            // write out the cached word
            cache[0]  = word;
            cache_idx = 1;
        }

        #pragma omp parallel for
        for (int i = word_rem; i < int( n_suffixes ); i += SYMBOLS_PER_WORD)
        {
            // encode a word's worth of characters
            word_type word = 0u;

            const uint32 n_symbols = nvbio::min( SYMBOLS_PER_WORD, n_suffixes - i );

            for (uint32 j = 0; j < n_symbols; ++j)
            {
                const uint32       bit_idx = j * SYMBOL_SIZE;
                const uint32 symbol_offset = BIG_ENDIAN ? (WORD_SIZE - SYMBOL_SIZE - bit_idx) : bit_idx;
                const word_type     symbol = word_type(h_bwt[i + j]) << symbol_offset;

                // set bits
                word |= symbol;
            }

            // write out the word and advance word_idx
            const uint32 word_idx = (i - word_rem) / SYMBOLS_PER_WORD;

            cache[ cache_idx + word_idx ] = word;
        }

        // compute how many words we can actually write out
        const uint32 n_full_words = cache_idx + (n_suffixes - word_rem) / SYMBOLS_PER_WORD;

        // write out the cache buffer
        {
            const uint32 n_bytes   = uint32( sizeof(word_type) * n_full_words );
            const uint32 n_written = BWTWriter::bwt_write( n_bytes, &cache[0] );
            if (n_written != n_bytes)
                throw nvbio::runtime_error("FileBWTHandler::process() : bwt write failed! (%u/%u bytes written)", n_written, n_bytes);
        }

        // save the last (possibly partial) word (hence the +2 guardband)
        cache_word = cache[ n_full_words ];

        const uint32 n_found_dollars = dollars.extract(
            n_suffixes,
            h_bwt,
            d_bwt,
            h_suffixes,
            d_suffixes,
            d_indices );

        // and write the list to the output
        if (n_found_dollars)
        {
            const uint32 n_bytes   = uint32( sizeof(DollarRankMap::entry_type) * n_found_dollars );
            const uint32 n_written = BWTWriter::index_write( n_bytes, &dollars.found_dollars[0] );
            if (n_written != n_bytes)
                throw nvbio::runtime_error("FileBWTHandler::process() : index write failed! (%u/%u bytes written)", n_written, n_bytes);
        }

        // advance the offset
        offset += n_suffixes;
    }

    uint64                  offset;
    std::vector<word_type>  cache;
    word_type               cache_word;
    DollarRankMap           dollars;
};

/// A class to output the BWT to a ASCII file
///
template <typename BWTWriter>
struct ASCIIFileBWTHandler : public BaseBWTHandler, public BWTWriter
{
    /// constructor
    ///
    ASCIIFileBWTHandler() : offset(0) {}

    /// destructor
    ///
    virtual ~ASCIIFileBWTHandler() {}

    /// write header
    ///
    void write_header()
    {
        const char* magic = "#PRI\n";       // PRImary-ASCII
        BWTWriter::index_write( 5, magic );
    }

    /// process a batch of BWT symbols
    ///
    void process(
        const uint32  n_suffixes,
        const uint8*  h_bwt,
        const uint8*  d_bwt,
        const uint2*  h_suffixes,
        const uint2*  d_suffixes,
        const uint32* d_indices)
    {
        // write out the cache buffer
        priv::alloc_storage( ascii, n_suffixes + 1 );

        // convert to ASCII
        dna6_to_string( h_bwt, n_suffixes, &ascii[0] );
        {
            const uint32 n_bytes   = uint32( n_suffixes );
            const uint32 n_written = BWTWriter::bwt_write( n_bytes, &ascii[0] );
            if (n_written != n_bytes)
                throw nvbio::runtime_error("FileBWTHandler::process() : bwt write failed! (%u/%u bytes written)", n_written, n_bytes);
        }

        const uint32 n_found_dollars = dollars.extract(
            n_suffixes,
            h_bwt,
            d_bwt,
            h_suffixes,
            d_suffixes,
            d_indices );

        // and write the list to the output
        if (n_found_dollars)
        {
            //const uint32 n_bytes   = uint32( sizeof(DollarRankMap::entry_type) * n_found_dollars );
            //const uint32 n_written = BWTWriter::index_write( n_bytes, &dollars.found_dollars[0] );
            //if (n_written != n_bytes)
            //    throw nvbio::runtime_error("FileBWTHandler::process() : index write failed! (%u/%u bytes written)", n_written, n_bytes);

            // reserve enough storage to encode 2 very large numbers in base 10 (up to 15 digits), plus a space and a newline
            priv::alloc_storage( dollar_buffer, n_found_dollars * 32 );

            uint32 output_size = 0;

            for (uint32 i = 0; i < n_found_dollars; ++i)
            {
                char* buf = &dollar_buffer[ output_size ];

                const uint32 len1 = itoa( buf,            dollars.found_dollars[i].first );  buf[len1]            = ' ';
                const uint32 len2 = itoa( buf + len1 + 1, dollars.found_dollars[i].second ); buf[len1 + len2 + 1] = '\n';
                const uint32 len  = len1 + len2 + 2;

                output_size += len;
            }

            const uint32 n_bytes   = output_size;
            const uint32 n_written = BWTWriter::index_write( n_bytes, &dollar_buffer[0] );
            if (n_written != n_bytes)
                throw nvbio::runtime_error("FileBWTHandler::process() : index write failed! (%u/%u bytes written)", n_written, n_bytes);
        }

        // advance the offset
        offset += n_suffixes;
    }

    uint64                  offset;
    std::vector<char>       ascii;
    DollarRankMap           dollars;
    std::vector<char>       dollar_buffer;
};

/// A class to output the BWT to a binary file
///
struct RawBWTWriter
{
    /// constructor
    ///
    RawBWTWriter();

    /// destructor
    ///
    ~RawBWTWriter();

    void open(const char* output_name, const char* index_name);

    /// write to the bwt
    ///
    uint32 bwt_write(const uint32 n_bytes, const void* buffer);

    /// write to the index
    ///
    uint32 index_write(const uint32 n_bytes, const void* buffer);

    /// return whether the file is in a good state
    ///
    bool is_ok() const;

private:
    FILE*   output_file;
    FILE*   index_file;
};

/// A class to output the BWT to a gzipped binary file
///
struct BWTGZWriter
{
    /// constructor
    ///
    BWTGZWriter();

    /// destructor
    ///
    ~BWTGZWriter();

    void open(const char* output_name, const char* index_name, const char* compression);

    /// write to the bwt
    ///
    uint32 bwt_write(const uint32 n_bytes, const void* buffer);

    /// write to the index
    ///
    uint32 index_write(const uint32 n_bytes, const void* buffer);

    /// return whether the file is in a good state
    ///
    bool is_ok() const;

private:
    void*   output_file;
    void*   index_file;
};

// constructor
//
RawBWTWriter::RawBWTWriter() :
    output_file(NULL),
    index_file(NULL)
{}

// destructor
//
RawBWTWriter::~RawBWTWriter()
{
    fclose( output_file );
    fclose( index_file );
}

void RawBWTWriter::open(const char* output_name, const char* index_name)
{
    log_verbose(stderr,"  opening bwt file \"%s\"\n", output_name);
    log_verbose(stderr,"  opening index file \"%s\"\n", index_name);
    output_file = fopen( output_name, "wb" );
    index_file  = fopen( index_name,  "wb" );
}

// write to the bwt
//
uint32 RawBWTWriter::bwt_write(const uint32 n_bytes, const void* buffer)
{
    return fwrite( buffer, sizeof(uint8), n_bytes, output_file );
}

// write to the index
//
uint32 RawBWTWriter::index_write(const uint32 n_bytes, const void* buffer)
{
    return fwrite( buffer, sizeof(uint8), n_bytes, index_file );
}

// return whether the file is in a good state
//
bool RawBWTWriter::is_ok() const { return output_file != NULL || index_file != NULL; }


// constructor
//
BWTGZWriter::BWTGZWriter() :
    output_file(NULL),
    index_file(NULL)
{}

// destructor
//
BWTGZWriter::~BWTGZWriter()
{
    gzclose( output_file );
    gzclose( index_file );
}

void BWTGZWriter::open(const char* output_name, const char* index_name, const char* compression)
{
    char comp_string[5];
    sprintf( comp_string, "wb%s", compression );

    log_verbose(stderr,"  opening bwt file \"%s\" (compression level: %s)\n", output_name, compression);
    log_verbose(stderr,"  opening index file \"%s\" (compression level: %s)\n", index_name, compression);
    output_file = gzopen( output_name, comp_string );
    index_file  = gzopen( index_name,  comp_string );
}

// write to the bwt
//
uint32 BWTGZWriter::bwt_write(const uint32 n_bytes, const void* buffer)
{
    return gzwrite( output_file, buffer, n_bytes );
}

// write to the index
//
uint32 BWTGZWriter::index_write(const uint32 n_bytes, const void* buffer)
{
    return gzwrite( index_file, buffer, n_bytes );
}

// return whether the file is in a good state
//
bool BWTGZWriter::is_ok() const { return output_file != NULL || index_file != NULL; }


// open a BWT file
//
BaseBWTHandler* open_bwt_file(const char* output_name, const char* params)
{
    enum OutputFormat
    {
        UNKNOWN = 0,
        TXT     = 1,
        TXTGZ   = 2,
        TXTBGZ  = 3,
        TXTLZ4  = 4,
        BWT2    = 5,
        BWT2GZ  = 6,
        BWT2BGZ = 7,
        BWT2LZ4 = 8,
        BWT4    = 9,
        BWT4GZ  = 10,
        BWT4BGZ = 11,
        BWT4LZ4 = 12,
    };
    OutputFormat format = UNKNOWN;
    std::string  index_string = output_name;

    // detect the file format from the suffix
    {
        const uint32 len = (uint32)strlen( output_name );

        //
        // detect BWT2* variants
        //
        if (len >= strlen(".bwt.bgz"))
        {
            if (strcmp(&output_name[len - strlen(".bwt.bgz")], ".bwt.bgz") == 0)
            {
                format = BWT2BGZ;
                index_string.replace( index_string.find(".bwt.bgz"), 8u, ".pri.bgz" );
            }
        }
        if (len >= strlen(".bwt.gz"))
        {
            if (strcmp(&output_name[len - strlen(".bwt.gz")], ".bwt.gz") == 0)
            {
                format = BWT2GZ;
                index_string.replace( index_string.find(".bwt.gz"), 7u, ".pri.gz" );
            }
        }
        if (len >= strlen(".bwt"))
        {
            if (strcmp(&output_name[len - strlen(".bwt")], ".bwt") == 0)
            {
                format = BWT2;
                index_string.replace( index_string.find(".bwt"), 4u, ".pri" );
            }
        }

        //
        // detect BWT4* variants
        //
        if (len >= strlen(".bwt4.bgz"))
        {
            if (strcmp(&output_name[len - strlen(".bwt4.bgz")], ".bwt4.bgz") == 0)
            {
                format = BWT4BGZ;
                index_string.replace( index_string.find(".bwt4.bgz"), 9u, ".pri.bgz" );
            }
        }
        if (len >= strlen(".bwt4.gz"))
        {
            if (strcmp(&output_name[len - strlen(".bwt4.gz")], ".bwt4.gz") == 0)
            {
                format = BWT4GZ;
                index_string.replace( index_string.find(".bwt4.gz"), 8u, ".pri.gz" );
            }
        }
        if (len >= strlen(".bwt4"))
        {
            if (strcmp(&output_name[len - strlen(".bwt4")], ".bwt4") == 0)
            {
                format = BWT4;
                index_string.replace( index_string.find(".bwt4"), 5u, ".pri" );
            }
        }

        //
        // detect TXT* variants
        //
        if (len >= strlen(".txt.gz"))
        {
            if (strcmp(&output_name[len - strlen(".txt.gz")], ".txt.gz") == 0)
            {
                format = TXTGZ;
                index_string.replace( index_string.find(".txt.gz"), 7u, ".pri.gz" );
            }
        }
        if (len >= strlen(".txt.bgz"))
        {
            if (strcmp(&output_name[len - strlen(".txt.bgz")], ".txt.bgz") == 0)
            {
                format = TXTGZ;
                index_string.replace( index_string.find(".txt.bgz"), 8u, ".pri.bgz" );
            }
        }
        if (len >= strlen(".txt"))
        {
            if (strcmp(&output_name[len - strlen(".txt")], ".txt") == 0)
            {
                format = TXT;
                index_string.replace( index_string.find(".txt"), 4u, ".pri" );
            }
        }
    }

    if (format == BWT2)
    {
        // build an output handler
        FileBWTHandler<RawBWTWriter,2,true,uint32>* file_handler = new FileBWTHandler<RawBWTWriter,2,true,uint32>();

        file_handler->open( output_name, index_string.c_str() );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == BWT2BGZ)
    {
        // build an output handler
        FileBWTHandler<BWTBGZWriter,2,true,uint32>* file_handler = new FileBWTHandler<BWTBGZWriter,2,true,uint32>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == BWT2GZ)
    {
        // build an output handler
        FileBWTHandler<BWTGZWriter,2,true,uint32>* file_handler = new FileBWTHandler<BWTGZWriter,2,true,uint32>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == BWT4)
    {
        // build an output handler
        FileBWTHandler<RawBWTWriter,4,true,uint32>* file_handler = new FileBWTHandler<RawBWTWriter,4,true,uint32>();

        file_handler->open( output_name, index_string.c_str() );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == BWT4BGZ)
    {
        // build an output handler
        FileBWTHandler<BWTBGZWriter,4,true,uint32>* file_handler = new FileBWTHandler<BWTBGZWriter,4,true,uint32>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == BWT4GZ)
    {
        // build an output handler
        FileBWTHandler<BWTGZWriter,4,true,uint32>* file_handler = new FileBWTHandler<BWTGZWriter,4,true,uint32>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == TXT)
    {
        // build an output handler
        ASCIIFileBWTHandler<RawBWTWriter>* file_handler = new ASCIIFileBWTHandler<RawBWTWriter>();

        file_handler->open( output_name, index_string.c_str() );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == TXTGZ)
    {
        // build an output handler
        ASCIIFileBWTHandler<BWTGZWriter>* file_handler = new ASCIIFileBWTHandler<BWTGZWriter>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }
    else if (format == TXTBGZ)
    {
        // build an output handler
        ASCIIFileBWTHandler<BWTBGZWriter>* file_handler = new ASCIIFileBWTHandler<BWTBGZWriter>();

        file_handler->open( output_name, index_string.c_str(), params );
        if (file_handler->is_ok() == false)
        {
            log_error(stderr,"  unable to open output file \"%s\"\n", output_name);
            return NULL;
        }
        file_handler->write_header();
        return file_handler;
    }

    log_error(stderr,"  unknown output format \"%s\"\n", output_name);
    return NULL;
}

} // namespace nvbio
