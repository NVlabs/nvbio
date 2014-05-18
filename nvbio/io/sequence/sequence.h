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

#include <nvbio/basic/strided_iterator.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/vector_view.h>
#include <nvbio/basic/vector.h>
#include <nvbio/strings/string_set.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace nvbio {

///
/// The supported sequence alphabet types
///
enum SequenceAlphabet
{
    DNA     = 0u,
    DNA_N   = 1u,
    PROTEIN = 2u
};

/// A traits class for SequenceAlphabet
///
template <SequenceAlphabet ALPHABET> struct SequenceAlphabetTraits {};

/// A traits class for DNA SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;
};
/// A traits class for DNA_N SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;
};
/// A traits class for Protein SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<PROTEIN>
{
    static const uint32 SYMBOL_SIZE  = 8;
    static const uint32 SYMBOL_COUNT = 24;
};

namespace io {

///
/// \page reads_io_page Sequence data input
/// This module contains a series of classes to load and represent read streams.
/// The idea is that a read stream is an object implementing a simple interface, \ref SequenceDataStream,
/// which allows to stream through a file or other set of reads in batches, which are represented in memory
/// with an object inheriting from SequenceData.
/// There are several kinds of SequenceData containers to keep the reads in the host RAM, or in CUDA device memory.
/// Additionally, the same container can be viewed with different SequenceDataView's, in order to allow reinterpreting
/// the base arrays as arrays of different types, e.g. to perform vector loads or use LDG.
///
/// Specifically, it exposes the following classes and methods:
///
/// - SequenceData
/// - SequenceDataView
/// - SequenceDataStream
/// - open_sequence_file()
///

///@addtogroup IO
///@{

///
///@defgroup SequencesIO Sequence data input
/// This module contains a series of classes to load and represent read streams.
/// The idea is that a read stream is an object implementing a simple interface, \ref SequenceDataStream,
/// which allows to stream through a file or other set of reads in batches, which are represented in memory
/// with an object inheriting from SequenceData.
/// There are several kinds of SequenceData containers to keep the reads in the host RAM, or in CUDA device memory.
/// Additionally, the same container can be viewed with different SequenceDataView's, in order to allow reinterpreting
/// the base arrays as arrays of different types, e.g. to perform vector loads or use LDG.
///@{
///

// describes the quality encoding for a given read file
enum QualityEncoding
{
    // phred quality
    Phred = 0,
    // phred quality + 33
    Phred33 = 1,
    // phred quality + 64
    Phred64 = 2,
    Solexa = 3,
};

// a set of flags describing the types of supported read strands
enum SequenceEncoding
{
    FORWARD            = 0x0001,
    REVERSE            = 0x0002,
    FORWARD_COMPLEMENT = 0x0004,
    REVERSE_COMPLEMENT = 0x0008,
};

// how mates of a paired-end read are encoded
// F = forward, R = reverse
enum PairedEndPolicy
{
    PE_POLICY_FF = 0,
    PE_POLICY_FR = 1,
    PE_POLICY_RF = 2,
    PE_POLICY_RR = 3,
};

///
/// A POD type encapsulating basic sequence information
///
struct SequenceDataInfo
{
    /// empty constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataInfo()
      : m_n_seqs(0),
        m_name_stream_len(0),
        m_sequence_stream_len(0),
        m_sequence_stream_words(0),
        m_min_sequence_len(uint32(-1)),
        m_max_sequence_len(0),
        m_avg_sequence_len(0)
    {};

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  size()                    const { return m_n_seqs; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  bps()                     const { return m_sequence_stream_len; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  words()                   const { return m_sequence_stream_words; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  name_stream_len()         const { return m_name_stream_len; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  max_sequence_len()        const { return m_max_sequence_len; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  min_sequence_len()        const { return m_min_sequence_len; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint32  avg_sequence_len()        const { return m_avg_sequence_len; }

    uint32  m_n_seqs;                   ///< number of reads in this struct
    uint32  m_name_stream_len;          ///< the length (in bytes) of the name_stream buffer
    uint32  m_sequence_stream_len;      ///< the length of sequence_stream in base pairs
    uint32  m_sequence_stream_words;    ///< the number of words in sequence_stream

    uint32  m_min_sequence_len;         ///< statistics on the reads
    uint32  m_max_sequence_len;         ///< statistics on the reads
    uint32  m_avg_sequence_len;         ///< statistics on the reads
};

template <SequenceAlphabet SEQUENCE_ALPHABET>
struct SequenceDataTraits
{
    // symbol size for reads
    static const uint32 SEQUENCE_BITS = SequenceAlphabetTraits<SEQUENCE_ALPHABET>::SYMBOL_SIZE;
    // big endian?
    static const bool   SEQUENCE_BIG_ENDIAN = false;
    // symbols per word
    static const uint32 SEQUENCE_SYMBOLS_PER_WORD = (4*sizeof(uint32))/SEQUENCE_BITS;
};

///
/// A storage-less plain-view class to represent and access sequence data.
///
/// This class is templated over the iterators pointing to the actual storage, so as to allow
/// them being both raw (const or non-const) pointers or fancier iterators (e.g. cuda::load_pointer
/// or nvbio::vector<system_tag>::iterator's)
///
/// \tparam IndexIterator               the type of the iterator to the reads index
/// \tparam ReadStorageIterator         the type of the iterator to the reads storage
/// \tparam QualStorageIterator         the type of the iterator to the qualities storage
/// \tparam NameStorageIterator         the type of the iterator to the names storage
///
template <
    SequenceAlphabet    SEQUENCE_ALPHABET_T,
    typename            IndexIterator               = uint32*,
    typename            SequenceStorageIterator     = uint32*,
    typename            QualStorageIterator         = char*,
    typename            NameStorageIterator         = char*>
struct SequenceDataView : public SequenceDataInfo
{
    static const SequenceAlphabet SEQUENCE_ALPHABET = SEQUENCE_ALPHABET_T;                                              ///< alphabet type
    static const uint32 SEQUENCE_BITS = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_BITS;                           ///< symbol size
    static const bool   SEQUENCE_BIG_ENDIAN = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_BIG_ENDIAN;               ///< endianness
    static const uint32 SEQUENCE_SYMBOLS_PER_WORD = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_SYMBOLS_PER_WORD;   ///< number of symbols per word

    typedef IndexIterator                                                             index_iterator;               ///< the index iterator
    typedef typename to_const<index_iterator>::type                             const_index_iterator;               ///< the const index iterator

    typedef SequenceStorageIterator                                                   sequence_storage_iterator;    ///< the read storage iterator
    typedef typename to_const<sequence_storage_iterator>::type                  const_sequence_storage_iterator;    ///< the const read storage iterator

    typedef QualStorageIterator                                                       qual_storage_iterator;        ///< the qualities iterator
    typedef typename to_const<qual_storage_iterator>::type                      const_qual_storage_iterator;        ///< the const qualities iterator

    typedef NameStorageIterator                                                       name_storage_iterator;        ///< the names string iterator
    typedef typename to_const<name_storage_iterator>::type                      const_name_storage_iterator;        ///< the names string iterator

    typedef PackedStream<
        sequence_storage_iterator,uint8,SEQUENCE_BITS,SEQUENCE_BIG_ENDIAN>              sequence_stream_type;       ///< the packed read-stream type
    typedef PackedStream<
        const_sequence_storage_iterator,uint8,SEQUENCE_BITS,SEQUENCE_BIG_ENDIAN>  const_sequence_stream_type;       ///< the const packed read-stream type

    typedef vector_view<sequence_stream_type>                                         sequence_string;            ///< the read string type
    typedef vector_view<const_sequence_stream_type>                             const_sequence_string;            ///< the const read string type

    typedef ConcatenatedStringSet<
        sequence_stream_type,
        index_iterator>                                                         sequence_string_set_type;   ///< string-set type

    typedef ConcatenatedStringSet<
        const_sequence_stream_type,
        const_index_iterator>                                             const_sequence_string_set_type;   ///< const string-set type

    typedef ConcatenatedStringSet<
        qual_storage_iterator,
        index_iterator>                                                         qual_string_set_type;   ///< quality string-set type

    typedef ConcatenatedStringSet<
        const_qual_storage_iterator,
        const_index_iterator>                                             const_qual_string_set_type;   ///< const quality string-set type

    typedef ConcatenatedStringSet<
        name_storage_iterator,
        index_iterator>                                                         name_string_set_type;   ///< name string-set type

    typedef ConcatenatedStringSet<
        const_name_storage_iterator,
        const_index_iterator>                                             const_name_string_set_type;   ///< const name string-set type

    /// empty constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataView() : SequenceDataInfo() {}

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataView(
        const SequenceDataInfo&         info,
        const SequenceStorageIterator   sequence_stream,
        const IndexIterator             sequence_index,
        const QualStorageIterator       qual_stream,
        const NameStorageIterator       name_stream,
        const IndexIterator             name_index)
      : SequenceDataInfo        ( info ),
        m_name_stream           (NameStorageIterator( name_stream )),
        m_name_index            (IndexIterator( name_index )),
        m_sequence_stream       (SequenceStorageIterator( sequence_stream )),
        m_sequence_index        (IndexIterator( sequence_index )),
        m_qual_stream           (QualStorageIterator( qual_stream ))
    {}
/*
    /// constructor
    ///
    template <
        typename InIndexIterator,
        typename InSequenceIterator,
        typename InQualIterator,
        typename InNameIterator>
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataView(
        const SequenceDataInfo&     info,
        const InSequenceIterator    sequence_stream,
        const IndexIterator         sequence_index,
        const InQualIterator        qual_stream,
        const InNameIterator        name_stream,
        const IndexIterator         name_index)
      : SequenceDataInfo        ( info ),
        m_name_stream           (NameStorageIterator( name_stream )),
        m_name_index            (IndexIterator( name_index )),
        m_sequence_stream       (SequenceStorageIterator( sequence_stream )),
        m_sequence_index        (IndexIterator( sequence_index )),
        m_qual_stream           (QualStorageIterator( qual_stream ))
    {}
*/
    /// copy constructor
    ///
    template <
        typename InIndexIterator,
        typename InSequenceIterator,
        typename InQualIterator,
        typename InNameIterator>
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataView(const SequenceDataView<SEQUENCE_ALPHABET_T,InIndexIterator,InSequenceIterator,InQualIterator,InNameIterator>& in)
      : SequenceDataInfo        ( in ),
        m_name_stream           (NameStorageIterator( in.m_name_stream )),
        m_name_index            (IndexIterator( in.m_name_index )),
        m_sequence_stream       (SequenceStorageIterator( in.m_sequence_stream )),
        m_sequence_index        (IndexIterator( in.m_sequence_index )),
        m_qual_stream           (QualStorageIterator( in.m_qual_stream ))
    {}

    /// assignment operator
    ///
    template <
        typename InIndexIterator,
        typename InSequenceIterator,
        typename InQualIterator,
        typename InNameIterator>
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SequenceDataView& operator=(const SequenceDataView<SEQUENCE_ALPHABET_T,InIndexIterator,InSequenceIterator,InQualIterator,InNameIterator>& in)
    {
        // copy the info
        this->SequenceDataInfo::operator=( in );

        // copy the iterators
        m_name_stream       = NameStorageIterator( in.m_name_stream );
        m_name_index        = IndexIterator( in.m_name_index );
        m_sequence_stream   = SequenceStorageIterator( in.m_sequence_stream );
        m_sequence_index    = IndexIterator( in.m_sequence_index );
        m_qual_stream       = QualStorageIterator( in.m_qual_stream );
    }

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE index_iterator              name_index()                { return m_name_index;  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE index_iterator              sequence_index()            { return m_sequence_index;  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE name_storage_iterator       name_stream()               { return m_name_stream; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE sequence_storage_iterator   sequence_stream_storage()   { return m_sequence_stream; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE sequence_stream_type        sequence_stream()           { return sequence_stream_type( m_sequence_stream ); }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE qual_storage_iterator       qual_stream()               { return m_qual_stream; }

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_index_iterator            const_name_index()              const { return m_name_index;  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_index_iterator            const_sequence_index()          const { return m_sequence_index;  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_name_storage_iterator     const_name_stream()             const { return m_name_stream; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_storage_iterator const_sequence_stream_storage() const { return m_sequence_stream; }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_stream_type      const_sequence_stream()         const { return const_sequence_stream_type( m_sequence_stream ); }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_qual_storage_iterator     const_qual_stream()             const { return m_qual_stream; }

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_index_iterator            name_index()                    const { return const_name_index();  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_index_iterator            sequence_index()                const { return const_sequence_index();  }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_name_storage_iterator     name_stream()                   const { return const_name_stream(); }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_storage_iterator sequence_stream_storage()       const { return const_sequence_stream_storage(); }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_stream_type      sequence_stream()               const { return const_sequence_stream(); }
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_qual_storage_iterator     qual_stream()                   const { return const_qual_stream(); }

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE uint2 get_range(const uint32 i) const { return make_uint2(m_sequence_index[i],m_sequence_index[i+1]); }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE sequence_string_set_type sequence_string_set()
    {
        return sequence_string_set_type(
            size(),
            sequence_stream().begin(),
            sequence_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_string_set_type sequence_string_set() const
    {
        return const_sequence_string_set_type(
            size(),
            sequence_stream().begin(),
            sequence_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_sequence_string_set_type const_sequence_string_set() const
    {
        return const_sequence_string_set_type(
            size(),
            sequence_stream().begin(),
            sequence_index() );
    }

    /// return the i-th read as a string
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE sequence_string get_read(const uint32 i)
    {
        const uint2 sequence_range = get_range( i );
        return sequence_string( sequence_range.y - sequence_range.x, sequence_stream().begin() + sequence_range.x );
    }

    /// return the i-th read as a string
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE sequence_string get_read(const uint32 i) const
    {
        const uint2 sequence_range = get_range( i );
        return const_sequence_string( sequence_range.y - sequence_range.x, sequence_stream().begin() + sequence_range.x );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE qual_string_set_type qual_string_set()
    {
        return qual_string_set_type(
            size(),
            qual_stream(),
            sequence_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_qual_string_set_type qual_string_set() const
    {
        return const_qual_string_set_type(
            size(),
            qual_stream(),
            sequence_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_qual_string_set_type const_qual_string_set() const
    {
        return const_qual_string_set_type(
            size(),
            qual_stream(),
            sequence_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE name_string_set_type name_string_set()
    {
        return name_string_set_type(
            size(),
            name_stream(),
            name_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_name_string_set_type name_string_set() const
    {
        return const_name_string_set_type(
            size(),
            name_stream(),
            name_index() );
    }

    /// return the a string-set view of this set of reads
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE const_name_string_set_type const_name_string_set() const
    {
        return const_name_string_set_type(
            size(),
            name_stream(),
            name_index() );
    }

public:
    // a pointer to a buffer containing the names of all the reads in this batch
    name_storage_iterator       m_name_stream;

    // an array of uint32 with the byte indices of the starting locations of each name in name_stream
    index_iterator              m_name_index;

    // a pointer to a buffer containing the read data
    // note that this could point at either host or device memory
    sequence_storage_iterator   m_sequence_stream;

    // an array of uint32 with the indices of the starting locations of each read in sequence_stream (in base pairs)
    index_iterator              m_sequence_index;

    // a pointer to a buffer containing quality data
    // (the indices in m_sequence_index are also valid for this buffer)
    qual_storage_iterator       m_qual_stream;
};

///
/// Base abstract class to encapsulate a sequence data object.
/// This class is meant to be a base for either host, shared or device memory objects
///
template <SequenceAlphabet SEQUENCE_ALPHABET_T>
struct SequenceData : public SequenceDataInfo
{
    static const SequenceAlphabet SEQUENCE_ALPHABET = SEQUENCE_ALPHABET_T;                                              ///< alphabet type
    static const uint32 SEQUENCE_BITS = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_BITS;                           ///< symbol size
    static const bool   SEQUENCE_BIG_ENDIAN = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_BIG_ENDIAN;               ///< endianness
    static const uint32 SEQUENCE_SYMBOLS_PER_WORD = SequenceDataTraits<SEQUENCE_ALPHABET>::SEQUENCE_SYMBOLS_PER_WORD;   ///< number of symbols per word

    typedef SequenceDataView<SEQUENCE_ALPHABET,uint32*,uint32*,char*,char*>                               plain_view_type;
    typedef SequenceDataView<SEQUENCE_ALPHABET,const uint32*,const uint32*,const char*,const char*> const_plain_view_type;

    /// virtual destructor
    ///
    virtual ~SequenceData() {}

    /// convert to a plain_view
    ///
    virtual operator       plain_view_type() = 0;

    /// convert to a const plain_view
    ///
    virtual operator const_plain_view_type() const = 0;
};

///
/// A concrete SequenceData storage implementation in host/device memory
///
template <typename system_tag, SequenceAlphabet SEQUENCE_ALPHABET>
struct SequenceDataStorage : public SequenceData<SEQUENCE_ALPHABET>
{
    typedef SequenceData<SEQUENCE_ALPHABET>                             SequenceDataBase;

    typedef typename SequenceDataBase::plain_view_type                  plain_view_type;
    typedef typename SequenceDataBase::const_plain_view_type      const_plain_view_type;

    /// assignment operator
    ///
    template <typename other_tag>
    SequenceDataStorage& operator= (const SequenceDataStorage<other_tag,SEQUENCE_ALPHABET>& other)
    {
        // copy the info
        this->SequenceDataInfo::operator=( other );

        // copy the vectors
        thrust_copy_vector( m_sequence_vec,       other.m_sequence_vec );
        thrust_copy_vector( m_sequence_index_vec, other.m_sequence_index_vec );
        thrust_copy_vector( m_qual_vec,           other.m_qual_vec );
        thrust_copy_vector( m_name_vec,           other.m_name_vec );
        thrust_copy_vector( m_name_index_vec,     other.m_name_index_vec );
        return *this;
    }

    /// convert to a plain_view
    ///
    operator plain_view_type()
    {
        return plain_view_type(
            static_cast<const SequenceDataInfo&>( *this ),
            nvbio::raw_pointer( m_sequence_vec ),
            nvbio::raw_pointer( m_sequence_index_vec ),
            nvbio::raw_pointer( m_qual_vec ),
            nvbio::raw_pointer( m_name_vec ),
            nvbio::raw_pointer( m_name_index_vec ) );
    }
    /// convert to a const plain_view
    ///
    operator const_plain_view_type() const
    {
        return const_plain_view_type(
            static_cast<const SequenceDataInfo&>( *this ),
            nvbio::raw_pointer( m_sequence_vec ),
            nvbio::raw_pointer( m_sequence_index_vec ),
            nvbio::raw_pointer( m_qual_vec ),
            nvbio::raw_pointer( m_name_vec ),
            nvbio::raw_pointer( m_name_index_vec ) );
    }

    // reserve enough storage for a given number of reads and bps
    //
    void reserve(const uint32 n_seqs, const uint32 n_bps)
    {
        // a default read id length used to reserve enough space upfront and avoid frequent allocations
        const uint32 AVG_NAME_LENGTH = 250;

        const uint32 bps_per_word = 32u / SEQUENCE_BITS;

        m_sequence_index_vec.reserve( n_seqs+1 );
        m_sequence_vec.reserve( n_bps / bps_per_word );
        m_qual_vec.reserve( n_bps );
        m_name_index_vec.reserve( AVG_NAME_LENGTH * n_seqs );
        m_name_index_vec.reserve( n_seqs+1 );
    }

    nvbio::vector<system_tag,uint32> m_sequence_vec;
    nvbio::vector<system_tag,uint32> m_sequence_index_vec;
    nvbio::vector<system_tag,char>   m_qual_vec;
    nvbio::vector<system_tag,char>   m_name_vec;
    nvbio::vector<system_tag,uint32> m_name_index_vec;
};

///
/// A host memory sequence-data object
///
template <SequenceAlphabet SEQUENCE_ALPHABET>
struct SequenceDataHost : public SequenceDataStorage<host_tag,SEQUENCE_ALPHABET> {};

///
/// A device memory sequence-data object
///
template <SequenceAlphabet SEQUENCE_ALPHABET>
struct SequenceDataDevice : public SequenceDataStorage<device_tag,SEQUENCE_ALPHABET> {};

///
/// A stream of SequenceData, allowing to process the associated reads in batches.
///
struct SequenceDataStream
{
    /// virtual destructor
    ///
    virtual ~SequenceDataStream() {}

    /// next batch
    ///
    bool next(struct SequenceDataEncoder* encoder, const uint32 batch_size, const uint32 batch_bps = uint32(-1)) {}

    /// is the stream ok?
    ///
    virtual bool is_ok() = 0;
};



/// factory method to open a read file
///
/// \param alphabet             the alphabet used to encode the read sequence
/// \param sequence_file_name   the file to open
/// \param qualities            the encoding of the qualities
/// \param max_seqs            maximum number of reads to input
/// \param max_sequence_len     maximum read length - reads will be truncated
/// \param flags                a set of flags indicating which strands to encode
///                             in the batch for each read.
///                             For example, passing FORWARD | REVERSE_COMPLEMENT
///                             will result in a stream containing BOTH the forward
///                             and reverse-complemented strands.
///
SequenceDataStream *open_sequence_file(const SequenceAlphabet   alphabet,
                                       const char *             sequence_file_name,
                                       const QualityEncoding    qualities,
                                       const uint32             max_seqs = uint32(-1),
                                       const uint32             max_sequence_len = uint32(-1),
                                       const SequenceEncoding   flags = REVERSE);

///@} // SequencesIO
///@} // IO

} // namespace io

/// return a plain view of a SequenceData object
///
template <SequenceAlphabet SEQUENCE_ALPHABET>
typename io::SequenceData<SEQUENCE_ALPHABET>::plain_view_type plain_view(io::SequenceData<SEQUENCE_ALPHABET>& sequence_data)
{
    return typename io::SequenceData<SEQUENCE_ALPHABET>::plain_view_type( sequence_data );
}

/// return a plain view of a const SequenceData object
///
template <SequenceAlphabet SEQUENCE_ALPHABET>
typename io::SequenceData<SEQUENCE_ALPHABET>::const_plain_view_type plain_view(const io::SequenceData<SEQUENCE_ALPHABET>& sequence_data)
{
    return typename io::SequenceData<SEQUENCE_ALPHABET>::const_plain_view_type( sequence_data );
}

} // namespace nvbio
