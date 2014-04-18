#include <thrust/sort.h>

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>

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

    mem_state::mem_vector_type mems(command_line_options.mems_batch);

    // reset the filter
    mem->mem_filter = mem_state::mem_filter_type();

    // search for MEMs
    mem->mem_filter.rank(THRESHOLD_KMEM_SEARCH, mem->f_index, mem->r_index, batch->const_read_string_set(),
                         command_line_options.min_intv, command_line_options.max_intv, command_line_options.min_span);

    log_info(stderr, "%.1f average ranges\n", float(mem->mem_filter.n_ranges()) / float(n_reads));
    log_info(stderr, "%.1f average MEMs\n", float(mem->mem_filter.n_mems()) / float(n_reads));

    // sort by read-id and range size
    thrust::sort(mems.begin(), mems.end(), read_id_sort());
}

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
