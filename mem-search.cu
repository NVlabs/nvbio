#include <thrust/sort.h>

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/priority_queue.h>
#include <nvbio/basic/transform_iterator.h>
#include <nvbio/basic/vector_wrapper.h>
#include <nvbio/basic/cuda/ldg.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "options.h"
#include "pipeline.h"
#include "mem-search.h"

using namespace nvbio;

static io::FMIndexData *load_genome(const char *name, uint32 flags)
{
    if (command_line_options.genome_use_mmap)
    {
        // try loading via mmap first
        io::FMIndexDataMMAP *mmap_loader = new io::FMIndexDataMMAP();
        if (mmap_loader->load(name))
        {
            return mmap_loader;
        }

        delete mmap_loader;
    }

    // fall back to file name
    io::FMIndexDataRAM *file_loader = new io::FMIndexDataRAM();
    if (!file_loader->load(name, flags))
    {
        log_error(stderr, "could not load genome %s\n");
        exit(1);
    }

    return file_loader;
}

void mem_init(struct pipeline_context *pipeline)
{
    // this specifies which portions of the FM index data to load
    const uint32 fm_flags = io::FMIndexData::GENOME  |
                            io::FMIndexData::FORWARD |
                            io::FMIndexData::REVERSE |
                            io::FMIndexData::SA;

    // load the genome on the host
    pipeline->mem.fmindex_data_host = load_genome(command_line_options.genome_file_name, fm_flags);
    // copy genome data to device
    pipeline->mem.fmindex_data_device = new io::FMIndexDataDevice(*pipeline->mem.fmindex_data_host, fm_flags);

    pipeline->mem.f_index = pipeline->mem.fmindex_data_device->index();
    pipeline->mem.r_index = pipeline->mem.fmindex_data_device->rindex();
}

// search MEMs for all reads in batch
void mem_search(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;
    const uint32 n_reads = batch->size();

    // reset the filter
    mem->mem_filter = mem_state::mem_filter_type();

    // search for MEMs
    mem->mem_filter.rank(THRESHOLD_KMEM_SEARCH, mem->f_index, mem->r_index, batch->const_read_string_set(),
                         command_line_options.min_intv, command_line_options.max_intv, command_line_options.min_span);

    log_info(stderr, "%.1f average ranges\n", float(mem->mem_filter.n_ranges()) / float(n_reads));
    log_info(stderr, "%.1f average MEMs\n", float(mem->mem_filter.n_mems()) / float(n_reads));
}

namespace {

#if defined(NVBIO_DEVICE_COMPILATION)
  #define HOST_DEVICE_STATEMENT( host, device ) device
#else
  #define HOST_DEVICE_STATEMENT( host, device ) host
#endif

// a functor to count the number of MEM hits produced by a given range of reads
struct hit_count_functor
{
    typedef uint32 argument_type;
    typedef uint32 result_type;

    // constructor
    hit_count_functor(struct mem_state* _mem) : mem( _mem ) {}

    // return the number of hits up to a given read
    NVBIO_HOST_DEVICE                                   // silence nvcc - this function is host only
    uint32 operator() (const uint32 read_id) const
    {
        return HOST_DEVICE_STATEMENT( mem->mem_filter.first_hit( read_id ), 0u );
    }

    struct mem_state* mem;
};

};

// given the first read in a chunk, determine a suitably sized chunk of reads
// (for which we can locate all MEMs in one go), updating pipeline::chunk
void fit_read_chunk(
    struct pipeline_context     *pipeline,
    const io::ReadDataDevice    *batch,
    const uint32                read_begin)     // first read in the chunk
{
    struct mem_state *mem = &pipeline->mem;

    const uint32 max_hits = command_line_options.mems_batch;

    //
    // use a binary search to locate the ending read forming a chunk with up to max_hits
    //

    pipeline->chunk.read_begin = read_begin;

    // determine the index of the first hit in the chunk
    pipeline->chunk.mem_begin = mem->mem_filter.first_hit( read_begin );

    // make an iterator to count the number of hits in a given range of reads
    typedef thrust::counting_iterator<uint32>                            index_iterator;
    typedef thrust::transform_iterator<hit_count_functor,index_iterator> hit_counting_iterator;

    const hit_counting_iterator hit_counter(
        thrust::make_counting_iterator( 0u ),
        hit_count_functor( mem ) );

    // perform the binary search
    pipeline->chunk.read_end = uint32( nvbio::upper_bound(
            max_hits,
            hit_counter + read_begin,
            batch->size() - read_begin ) - hit_counter );

    // determine the index of the ending hit in the chunk
    pipeline->chunk.mem_end = mem->mem_filter.first_hit( pipeline->chunk.read_end );
}

// a functor to extract the reference location from a mem
struct mem_loc_functor
{
    typedef mem_state::mem_type argument_type;
    typedef uint64              result_type;

    NVBIO_HOST_DEVICE
    uint64 operator() (const argument_type mem) const
    {
        return uint64( mem.index_pos() ) | (uint64( mem.string_id() ) << 32);
    }
};

// a functor to extract the read id from a mem
struct mem_read_id_functor
{
    typedef mem_state::mem_type argument_type;
    typedef uint32              result_type;

    NVBIO_HOST_DEVICE
    uint32 operator() (const argument_type mem) const { return mem.string_id(); }
};

// locate all mems in the range defined by pipeline::chunk
void mem_locate(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;

    if (mem->mems.size() < command_line_options.mems_batch)
    {
        mem->mems.resize( command_line_options.mems_batch );
        mem->mems_index.resize( command_line_options.mems_batch );
    }

    const uint32 n_mems = pipeline->chunk.mem_end - pipeline->chunk.mem_begin;

    mem->mem_filter.locate(
        pipeline->chunk.mem_begin,
        pipeline->chunk.mem_end,
        mem->mems.begin() );

    // sort the mems by reference location
    nvbio::vector<device_tag,uint64> loc( n_mems );

    thrust::transform(
        mem->mems.begin(),
        mem->mems.begin() + n_mems,
        loc.begin(),
        mem_loc_functor() );

    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_mems,
        mem->mems_index.begin() );

    // TODO: this is slow, switch to nvbio::cuda::SortEnactor
    thrust::sort_by_key(
        loc.begin(),
        loc.begin() + n_mems,
        mem->mems_index.begin() );
}

struct chain
{
    // construct an empty chain
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    chain() {}

    // construct a new chain from a single seed
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    chain(const mem_state::mem_type seed) : ref( seed.index_pos() ) {}

    // test whether we can merge the given mem into this chain
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool merge(const mem_state::mem_type seed)
    {
        return false;
    }

    // a list of seeds : how do we store it? we need a large arena of some kind from where
    // to carve list entries...
    //seed_list seeds;

    uint32 ref; // cache the leftmost reference coordinate
};

struct chain_compare
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool operator() (const chain& chain1, const chain& chain2) const
    {
        // compare by the reference coordinate of the first seed of each chain
        return chain1.ref < chain2.ref;
    }
};

// build chains for the current pipeline::chunk of reads
__global__
void build_chains_kernel(
    const io::ReadDataDevice::const_plain_view_type batch,
    const read_chunk                                chunk,
    const uint32                                    n_mems,
    const mem_state::mem_type*                      mems,
    const uint32*                                   mems_index)
{
    const uint32 read_id = threadIdx.x + blockIdx.x * blockDim.x + chunk.read_begin;
    if (read_id >= chunk.read_end)
        return;

    // find the first seed belonging to this read
    const uint32 mem_begin = uint32( nvbio::lower_bound(
        read_id,
        nvbio::make_transform_iterator( mems, mem_read_id_functor() ),
        n_mems ) - nvbio::make_transform_iterator( mems, mem_read_id_functor() ) );

    // find the first seed belonging to the next read
    const uint32 mem_end = uint32( nvbio::lower_bound(
        read_id+1u,
        nvbio::make_transform_iterator( mems, mem_read_id_functor() ),
        n_mems ) - nvbio::make_transform_iterator( mems, mem_read_id_functor() ) );

    typedef nvbio::vector_wrapper<chain*> chain_vector_type;
    const uint32 MAX_CHAINS = 2048;     // need to handle overflow in multiple passes...

    // keep a priority queue of the chains organized by the reference coordinate of their leftmost seed
    chain chain_queue_storage[MAX_CHAINS];
    nvbio::priority_queue<chain, chain_vector_type, chain_compare> chain_queue( chain_vector_type( 0u, chain_queue_storage ) );

    // process the seeds in order
    for (uint32 i = mem_begin; i < mem_end; ++i)
    {
        const mem_state::mem_type seed = mems[ mems_index[i] ];

        // insert seed
        if (chain_queue.empty())
        {
            // build a new chain
            chain_queue.push( chain(seed) );
        }
        else
        {
            // find the closest chain...
            chain& chn = chain_queue.top();

            // and test whether we can merge this seed into it
            if (chn.merge( seed ) == false)
                chain_queue.push( chain(seed) );
        }
    }
}

// build chains for the current pipeline::chunk of reads
void build_chains(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;

    const uint32 n_reads = pipeline->chunk.read_end - pipeline->chunk.read_begin;
    const uint32 n_mems  = pipeline->chunk.mem_end  - pipeline->chunk.mem_begin;

    const uint32 block_dim = 128;
    const uint32 n_blocks  = util::divide_ri( n_reads, block_dim );

    build_chains_kernel<<<n_blocks, block_dim>>>(
        nvbio::plain_view( *batch ),
        pipeline->chunk,
        n_mems,
        nvbio::plain_view( mem->mems ),
        nvbio::plain_view( mem->mems_index ) );
}
