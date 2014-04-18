#include <thrust/sort.h>

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
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

// utility class to sort MEMs by increasing read-id first, then by decreasing number of hits
class read_id_sort
{
public:
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE int operator() (mem_state::mem_filter_type::mem_type a, mem_state::mem_filter_type::mem_type b)
    {
        // x = SA-begin, y = SA-end, z = string-id, w = string_end<<16|string_begin
        if (a.z == b.z)
        {
            // for the same read-id, sort by decreasing SA range size
            return (a.y - a.x) > (b.y - b.x);
        } else {
            return a.z < b.z;
        }
    }
};

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

// locate all mems in the range defined by pipeline::chunk
void mem_locate(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;

    if (mem->mems.size() < command_line_options.mems_batch)
        mem->mems.resize( command_line_options.mems_batch );

    mem->mem_filter.locate(
        pipeline->chunk.mem_begin,
        pipeline->chunk.mem_end,
        mem->mems.begin() );
}

#if 0

#if 0 // this one seems like it'll be slower
__global__ void compute_read_boundaries(uint32 *read_start_offsets, uint32 *read_hits_scan, mem_state::mem_filter_type::mem_type *mems, uint64 *slots, uint32 n_hits, uint32 n_reads)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    read_start_offsets[0] = 0;
    read_hits_scan[0] = 0;

    if (tid > n_hits - 1)
        return;

    mem_state::mem_filter_type::mem_type a, b;
    a = mems[tid];
    b = mems[tid + 1];

    if (a.z != b.z)
    {
        read_start_offsets[b.z] = tid + 1;
        read_hits_scan[b.z] = slots[tid + 1];
    }
}
#else
struct is_read_boundary
{
    uint32 read;
    __device__ is_read_boundary(uint32 r) { read = r; }
    NVBIO_HOST_DEVICE bool operator() (const mem_state::mem_filter_type::mem_type& mem) const
    {
        return mem.z == read;
    }
};

// search for read boundaries in mems and keep track of the running sum of range sizes
__global__ void compute_read_boundaries(uint32 *read_start_offsets, uint32 *read_hits_scan, mem_state::mem_filter_type::mem_type *mems, uint64 *slots, uint32 n_hits, uint32 n_reads)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    read_start_offsets[0] = 0;
    read_hits_scan[0] = 0;

    if (tid > n_reads)
        return;

    if (tid == 0)
    {
        read_start_offsets[0] = 0;
        read_hits_scan[0] = 0;
        return;
    }

    mem_state::mem_filter_type::mem_type *boundary = nvbio::find_pivot(mems, n_hits, is_read_boundary(tid));
    read_start_offsets[tid] = boundary - mems;
    read_hits_scan[tid] = slots[boundary - mems];
}
#endif

// split the mems vector into chunks
// we want each chunk to contain ~16M hits, while keeping all hits for any given read inside the same chunk
void mem_split(struct pipeline_context *pipeline, const io::ReadDataDevice *batch)
{
    struct mem_state *mem = &pipeline->mem;
    const uint32 n_reads = batch->size();
    const uint32 n_hits = mem->mems.size();

    // read_start_offsets[i] is the i'th reads' starting offset in mems
    thrust::device_vector<uint32> read_start_offsets(n_reads);
    // read_hits_scan[i] is the prefix sum of all the hits up to the i'th read in the batch
    thrust::device_vector<uint32> read_hits_scan(n_reads);

    const uint32 block_dim = 1024;
#if 0
    const uint32 n_blocks = nvbio::util::divide_ri(n_hits, block_dim);
#else
    const uint32 n_blocks = nvbio::util::divide_ri(n_reads, block_dim);
#endif
    compute_read_boundaries<<<n_blocks, block_dim>>>(thrust::raw_pointer_cast(&read_start_offsets[0]),
                                                     thrust::raw_pointer_cast(&read_hits_scan[0]),
                                                     plain_view(mem->mems),
                                                     thrust::raw_pointer_cast(&mem->mem_filter.m_slots[0]),
                                                     n_hits, n_reads);
}

#endif