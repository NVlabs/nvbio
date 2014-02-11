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

namespace nvbio {
namespace bowtie2 {
namespace cuda {

// resize the arena
//
// \return     # of allocated bytes
inline
uint64 SeedHitDequeArray::resize(const uint32 n_reads, const uint32 max_hits, const bool do_alloc)
{
    uint64 bytes = 0u;
    uint32 max_nodes = SumTree<float*>::node_count( max_hits );
    if (do_alloc) m_counts.resize( n_reads );               bytes += n_reads * sizeof(uint32);
    if (do_alloc) m_index.resize( n_reads );                bytes += n_reads * sizeof(uint32);
    if (do_alloc) m_hits.resize( n_reads * max_hits );      bytes += n_reads * max_hits * sizeof(SeedHit);
    if (do_alloc) m_probs.resize( n_reads * max_nodes );    bytes += n_reads * max_nodes * sizeof(float);
    if (do_alloc) m_pool.resize( 1, 0u );                   bytes += sizeof(uint32);

    if (do_alloc) thrust::fill( m_counts.begin(), m_counts.end(), uint32(0) );
    if (do_alloc) thrust::fill( m_index.begin(),  m_index.end(),  uint32(0) );
    return bytes;
}

/// clear all deques
///
inline
void SeedHitDequeArray::clear_deques()
{
    // reset deque size counters
    thrust::fill( m_counts.begin(), m_counts.end(), uint32(0) );
    thrust::fill( m_index.begin(),  m_index.end(),  uint32(0) );

    m_pool[0] = 0; // reset the arena
}

// return the device view
//
inline
SeedHitDequeArrayDeviceView SeedHitDequeArray::device_view()
{
    return SeedHitDequeArrayDeviceView(
        nvbio::device_view( m_counts ),
        nvbio::device_view( m_index ),
        nvbio::device_view( m_hits ),
        nvbio::device_view( m_probs ),
        nvbio::device_view( m_pool ) );
}

// constructor
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
SeedHitDequeArrayDeviceView::SeedHitDequeArrayDeviceView(
    index_storage_type  counts,
    index_storage_type  index,
    hits_storage_type   hits,
    prob_storage_type   probs,
    index_storage_type  pool) :
    m_counts( counts ),
    m_hits( hits ),
    m_probs( probs ),
    m_index( index ),
    m_pool( pool )
{}

// return a reference to the given deque
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
SeedHitDequeArrayDeviceView::reference SeedHitDequeArrayDeviceView::operator[] (const uint32 read_id)
{
    return reference( *this, read_id );
}

// allocate some storage for the deque bound to a given read
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
SeedHit* SeedHitDequeArrayDeviceView::alloc_deque(const uint32 read_id, const uint32 size)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    m_counts[read_id] = 0u;
    m_index[read_id]  = size ? atomicAdd( m_pool, size ) : 0u;
    return m_hits + m_index[read_id];
#else
    return NULL;
#endif
}

// return the deque bound to a given read
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename SeedHitDequeArrayDeviceView::hit_deque_type SeedHitDequeArrayDeviceView::get_deque(const uint32 read_id, bool build_heap) const
{
    hit_vector_type hit_vector( m_counts[read_id], get_data( read_id ) );
    return hit_deque_type( hit_vector, build_heap );
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
