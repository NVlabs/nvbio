
/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
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
 *
 ******************************************************************************/

/**
 * \file
 * cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within global memory.
 */

#pragma once

#include <stdio.h>
#include <iterator>

#include "region/block_scan_region.cuh"
#include "../thread/thread_operators.cuh"
#include "../grid/grid_queue.cuh"
#include "../util_debug.cuh"
#include "../util_device.cuh"
#include "../util_namespace.cuh"

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/******************************************************************************
 * Kernel entry points
 *****************************************************************************/

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document


/**
 * Initialization kernel for tile status initialization (multi-block)
 */
template <
    typename T,                                     ///< Scan value type
    typename Offset>                                ///< Signed integer type for global offsets
__global__ void ScanInitKernel(
    GridQueue<Offset>           grid_queue,         ///< [in] Descriptor for performing dynamic mapping of input tiles to thread blocks
    LookbackTileDescriptor<T>   *d_tile_status,     ///< [out] Tile status words
    int                         num_tiles)          ///< [in] Number of tiles
{
    typedef LookbackTileDescriptor<T> TileDescriptor;

    enum
    {
        TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
    };

    // Reset queue descriptor
    if ((blockIdx.x == 0) && (threadIdx.x == 0)) grid_queue.FillAndResetDrain(num_tiles);

    // Initialize tile status
    int tile_offset = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tile_offset < num_tiles)
    {
        // Not-yet-set
        d_tile_status[TILE_STATUS_PADDING + tile_offset].status = LOOKBACK_TILE_INVALID;
    }

    if ((blockIdx.x == 0) && (threadIdx.x < TILE_STATUS_PADDING))
    {
        // Padding
        d_tile_status[threadIdx.x].status = LOOKBACK_TILE_OOB;
    }
}


/**
 * Scan kernel entry point (multi-block)
 */
template <
    typename    BlockScanRegionPolicy,          ///< Parameterized BlockScanRegionPolicy tuning policy type
    typename    InputIterator,                  ///< Random-access iterator type for input \iterator
    typename    OutputIterator,                 ///< Random-access iterator type for output \iterator
    typename    T,                              ///< The scan data type
    typename    ScanOp,                         ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename    Identity,                       ///< Identity value type (cub::NullType for inclusive scans)
    typename    Offset>                         ///< Signed integer type for global offsets
__launch_bounds__ (int(BlockScanRegionPolicy::BLOCK_THREADS))
__global__ void ScanRegionKernel(
    InputIterator               d_in,           ///< Input data
    OutputIterator              d_out,          ///< Output data
    LookbackTileDescriptor<T>   *d_tile_status, ///< Global list of tile status
    ScanOp                      scan_op,        ///< Binary scan operator
    Identity                    identity,       ///< Identity element
    Offset                      num_items,      ///< Total number of scan items for the entire problem
    GridQueue<int>              queue)          ///< Drain queue descriptor for dynamically mapping tile data onto thread blocks
{
    enum
    {
        TILE_STATUS_PADDING = CUB_PTX_WARP_THREADS,
    };

    // Thread block type for scanning input tiles
    typedef BlockScanRegion<
        BlockScanRegionPolicy,
        InputIterator,
        OutputIterator,
        ScanOp,
        Identity,
        Offset> BlockScanRegionT;

    // Shared memory for BlockScanRegion
    __shared__ typename BlockScanRegionT::TempStorage temp_storage;

    // Process tiles
    BlockScanRegionT(temp_storage, d_in, d_out, scan_op, identity).ConsumeRegion(
        num_items,
        queue,
        d_tile_status + TILE_STATUS_PADDING);
}




/******************************************************************************
 * Dispatch
 ******************************************************************************/

/**
 * Internal dispatch routine
 */
template <
    typename InputIterator,      ///< Random-access iterator type for input \iterator
    typename OutputIterator,     ///< Random-access iterator type for output \iterator
    typename ScanOp,             ///< Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
    typename Identity,           ///< Identity value type (cub::NullType for inclusive scans)
    typename Offset>             ///< Signed integer type for global offsets
struct DeviceScanDispatch
{
    enum
    {
        TILE_STATUS_PADDING     = 32,
        INIT_KERNEL_THREADS     = 128
    };

    // Data type
    typedef typename std::iterator_traits<InputIterator>::value_type T;

    // Tile status descriptor type
    typedef LookbackTileDescriptor<T> TileDescriptor;


    /******************************************************************************
     * Tuning policies
     ******************************************************************************/

    /// SM35
    struct Policy350
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 16,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // GTX Titan: 29.1B items/s (232.4 GB/s) @ 48M 32-bit T
        typedef BlockScanRegionPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_DIRECT,
                false,
                LOAD_LDG,
                BLOCK_STORE_WARP_TRANSPOSE,
                true,
                BLOCK_SCAN_RAKING_MEMOIZE>
            ScanRegionPolicy;
    };

    /// SM30
    struct Policy300
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 9,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanRegionPolicy<
                256,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                false,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
            ScanRegionPolicy;
    };

    /// SM20
    struct Policy200
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 15,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        // GTX 580: 20.3B items/s (162.3 GB/s) @ 48M 32-bit T
        typedef BlockScanRegionPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                false,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                false,
                BLOCK_SCAN_RAKING_MEMOIZE>
            ScanRegionPolicy;
    };

    /// SM13
    struct Policy130
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 19,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanRegionPolicy<
                64,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                true,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                true,
                BLOCK_SCAN_RAKING_MEMOIZE>
            ScanRegionPolicy;
    };

    /// SM10
    struct Policy100
    {
        enum {
            NOMINAL_4B_ITEMS_PER_THREAD = 19,
            ITEMS_PER_THREAD            = CUB_MIN(NOMINAL_4B_ITEMS_PER_THREAD, CUB_MAX(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / sizeof(T)))),
        };

        typedef BlockScanRegionPolicy<
                128,
                ITEMS_PER_THREAD,
                BLOCK_LOAD_WARP_TRANSPOSE,
                true,
                LOAD_DEFAULT,
                BLOCK_STORE_WARP_TRANSPOSE,
                true,
                BLOCK_SCAN_RAKING>
            ScanRegionPolicy;
    };


    /******************************************************************************
     * Tuning policies of current PTX compiler pass
     ******************************************************************************/

#if (CUB_PTX_VERSION >= 350)
    typedef Policy350 PtxPolicy;

#elif (CUB_PTX_VERSION >= 300)
    typedef Policy300 PtxPolicy;

#elif (CUB_PTX_VERSION >= 200)
    typedef Policy200 PtxPolicy;

#elif (CUB_PTX_VERSION >= 130)
    typedef Policy130 PtxPolicy;

#else
    typedef Policy100 PtxPolicy;

#endif

    // "Opaque" policies (whose parameterizations aren't reflected in the type signature)
    struct PtxScanRegionPolicy : PtxPolicy::ScanRegionPolicy {};


    /******************************************************************************
     * Utilities
     ******************************************************************************/

    /**
     * Initialize kernel dispatch configurations with the policies corresponding to the PTX assembly we will use
     */
    template <typename KernelConfig>
    __host__ __device__ __forceinline__
    static void InitConfigs(
        int             ptx_version,
        KernelConfig    &scan_region_config)
    {
    #ifdef __CUDA_ARCH__

        // We're on the device, so initialize the kernel dispatch configurations with the current PTX policy
        scan_region_config.template Init<PtxScanRegionPolicy>();

    #else

        // We're on the host, so lookup and initialize the kernel dispatch configurations with the policies that match the device's PTX version
        if (ptx_version >= 350)
        {
            scan_region_config.template Init<typename Policy350::ScanRegionPolicy>();
        }
        else if (ptx_version >= 300)
        {
            scan_region_config.template Init<typename Policy300::ScanRegionPolicy>();
        }
        else if (ptx_version >= 200)
        {
            scan_region_config.template Init<typename Policy200::ScanRegionPolicy>();
        }
        else if (ptx_version >= 130)
        {
            scan_region_config.template Init<typename Policy130::ScanRegionPolicy>();
        }
        else
        {
            scan_region_config.template Init<typename Policy100::ScanRegionPolicy>();
        }

    #endif
    }


    /**
     * Kernel kernel dispatch configuration.  Mirrors the constants within BlockScanRegionPolicy.
     */
    struct KernelConfig
    {
        int                     block_threads;
        int                     items_per_thread;
        BlockLoadAlgorithm      load_policy;
        BlockStoreAlgorithm     store_policy;
        BlockScanAlgorithm      scan_algorithm;

        template <typename BlockScanRegionPolicy>
        __host__ __device__ __forceinline__
        void Init()
        {
            block_threads               = BlockScanRegionPolicy::BLOCK_THREADS;
            items_per_thread            = BlockScanRegionPolicy::ITEMS_PER_THREAD;
            load_policy                 = BlockScanRegionPolicy::LOAD_ALGORITHM;
            store_policy                = BlockScanRegionPolicy::STORE_ALGORITHM;
            scan_algorithm              = BlockScanRegionPolicy::SCAN_ALGORITHM;
        }

        __host__ __device__ __forceinline__
        void Print()
        {
            printf("%d, %d, %d, %d, %d",
                block_threads,
                items_per_thread,
                load_policy,
                store_policy,
                scan_algorithm);
        }
    };


    /******************************************************************************
     * Dispatch entrypoints
     ******************************************************************************/

    /**
     * Internal dispatch routine for computing a device-wide prefix scan using the
     * specified kernel functions.
     */
    template <
        typename                    ScanInitKernelPtr,              ///< Function type of cub::ScanInitKernel
        typename                    ScanRegionKernelPtr>            ///< Function type of cub::ScanRegionKernelPtr
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void                        *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t                      &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator               d_in,                           ///< [in] Iterator pointing to scan input
        OutputIterator              d_out,                          ///< [in] Iterator pointing to scan output
        ScanOp                      scan_op,                        ///< [in] Binary scan operator
        Identity                    identity,                       ///< [in] Identity element
        Offset                      num_items,                      ///< [in] Total number of items to scan
        cudaStream_t                stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                        debug_synchronous,              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
        int                         ptx_version,                    ///< [in] PTX version of dispatch kernels
        ScanInitKernelPtr           init_kernel,                    ///< [in] Kernel function pointer to parameterization of cub::ScanInitKernel
        ScanRegionKernelPtr         scan_region_kernel,             ///< [in] Kernel function pointer to parameterization of cub::ScanRegionKernelPtr
        KernelConfig                scan_region_config)             ///< [in] Dispatch parameters that match the policy that \p scan_region_kernel was compiled for
    {

#ifndef CUB_RUNTIME_ENABLED

        // Kernel launch not supported from this device
        return CubDebug(cudaErrorNotSupported);

#else
        cudaError error = cudaSuccess;
        do
        {
            // Get device ordinal
            int device_ordinal;
            if (CubDebug(error = cudaGetDevice(&device_ordinal))) break;

            // Get device SM version
            int sm_version;
            if (CubDebug(error = SmVersion(sm_version, device_ordinal))) break;

            // Get SM count
            int sm_count;
            if (CubDebug(error = cudaDeviceGetAttribute (&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal))) break;

            // Number of input tiles
            int tile_size = scan_region_config.block_threads * scan_region_config.items_per_thread;
            int num_tiles = (num_items + tile_size - 1) / tile_size;

            // Temporary storage allocation requirements
            void* allocations[2];
            size_t allocation_sizes[2] =
            {
                (num_tiles + TILE_STATUS_PADDING) * sizeof(TileDescriptor),  // bytes needed for tile status descriptors
                GridQueue<int>::AllocationSize()                             // bytes needed for grid queue descriptor
            };

            // Alias the temporary allocations from the single storage blob (or set the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (d_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            // Alias the allocation for the global list of tile status
            TileDescriptor *d_tile_status = (TileDescriptor*) allocations[0];

            // Alias the allocation for the grid queue descriptor
            GridQueue<int> queue(allocations[1]);

            // Log init_kernel configuration
            int init_grid_size = (num_tiles + INIT_KERNEL_THREADS - 1) / INIT_KERNEL_THREADS;
            if (debug_synchronous) CubLog("Invoking init_kernel<<<%d, %d, 0, %lld>>>()\n", init_grid_size, INIT_KERNEL_THREADS, (long long) stream);

            // Invoke init_kernel to initialize tile descriptors and queue descriptors
            init_kernel<<<init_grid_size, INIT_KERNEL_THREADS, 0, stream>>>(
                queue,
                d_tile_status,
                num_tiles);

            // Sync the stream if specified
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;

            // Get SM occupancy for scan_region_kernel
            int scan_region_sm_occupancy;
            if (CubDebug(error = MaxSmOccupancy(
                scan_region_sm_occupancy,            // out
                sm_version,
                scan_region_kernel,
                scan_region_config.block_threads))) break;

            // Get device occupancy for scan_region_kernel
            int scan_region_occupancy = scan_region_sm_occupancy * sm_count;

            // Get grid size for scanning tiles
            int scan_grid_size;
            if (ptx_version < 200)
            {
                // We don't have atomics (or don't have fast ones), so just assign one block per tile (limited to 65K tiles)
                scan_grid_size = num_tiles;
                if (scan_grid_size >= (64 * 1024))
                    return cudaErrorInvalidConfiguration;
            }
            else
            {
                scan_grid_size = (num_tiles < scan_region_occupancy) ?
                    num_tiles :                     // Not enough to fill the device with threadblocks
                    scan_region_occupancy;          // Fill the device with threadblocks
            }

            // Log scan_region_kernel configuration
            if (debug_synchronous) CubLog("Invoking scan_region_kernel<<<%d, %d, 0, %lld>>>(), %d items per thread, %d SM occupancy\n",
                scan_grid_size, scan_region_config.block_threads, (long long) stream, scan_region_config.items_per_thread, scan_region_sm_occupancy);

            // Invoke scan_region_kernel
            scan_region_kernel<<<scan_grid_size, scan_region_config.block_threads, 0, stream>>>(
                d_in,
                d_out,
                d_tile_status,
                scan_op,
                identity,
                num_items,
                queue);

            // Sync the stream if specified
            if (debug_synchronous && (CubDebug(error = SyncStream(stream)))) break;
        }
        while (0);

        return error;

#endif  // CUB_RUNTIME_ENABLED
    }


    /**
     * Internal dispatch routine
     */
    __host__ __device__ __forceinline__
    static cudaError_t Dispatch(
        void            *d_temp_storage,                ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,            ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                           ///< [in] Iterator pointing to scan input
        OutputIterator  d_out,                          ///< [in] Iterator pointing to scan output
        ScanOp          scan_op,                        ///< [in] Binary scan operator
        Identity        identity,                       ///< [in] Identity element
        Offset          num_items,                      ///< [in] Total number of items to scan
        cudaStream_t    stream,                         ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous)              ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  Also causes launch configurations to be printed to the console.  Default is \p false.
    {
        cudaError error = cudaSuccess;
        do
        {
            // Get PTX version
            int ptx_version;
    #ifndef __CUDA_ARCH__
            if (CubDebug(error = PtxVersion(ptx_version))) break;
    #else
            ptx_version = CUB_PTX_VERSION;
    #endif

            // Get kernel kernel dispatch configurations
            KernelConfig scan_region_config;
            InitConfigs(ptx_version, scan_region_config);

            // Dispatch
            if (CubDebug(error = Dispatch(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                scan_op,
                identity,
                num_items,
                stream,
                debug_synchronous,
                ptx_version,
                ScanInitKernel<T, Offset>,
                ScanRegionKernel<PtxScanRegionPolicy, InputIterator, OutputIterator, T, ScanOp, Identity, Offset>,
                scan_region_config))) break;
        }
        while (0);

        return error;
    }
};



#endif // DOXYGEN_SHOULD_SKIP_THIS



/******************************************************************************
 * DeviceScan
 *****************************************************************************/

/**
 * \brief DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data items residing within global memory. ![](device_scan.png)
 * \ingroup DeviceModule
 *
 * \par Overview
 * Given a sequence of input elements and a binary reduction operator, a [<em>prefix scan</em>](http://en.wikipedia.org/wiki/Prefix_sum)
 * produces an output sequence where each element is computed to be the reduction
 * of the elements occurring earlier in the input sequence.  <em>Prefix sum</em>
 * connotes a prefix scan with the addition operator. The term \em inclusive indicates
 * that the <em>i</em><sup>th</sup> output reduction incorporates the <em>i</em><sup>th</sup> input.
 * The term \em exclusive indicates the <em>i</em><sup>th</sup> input is not incorporated into
 * the <em>i</em><sup>th</sup> output reduction.
 *
 * \par Usage Considerations
 * \cdp_class{DeviceScan}
 *
 * \par Performance
 *
 * \image html scan_perf.png
 *
 */
struct DeviceScan
{
    /******************************************************************//**
     * \name Exclusive scans
     *********************************************************************/
    //@{

    /**
     * \brief Computes a device-wide exclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the exclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix sum
     * cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out s<-- [0, 8, 14, 21, 26, 29, 29]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     */
    template <
        typename        InputIterator,
        typename        OutputIterator>
    __host__ __device__
    static cudaError_t ExclusiveSum(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Iterator pointing to scan input
        OutputIterator  d_out,                              ///< [in] Iterator pointing to scan output
        int             num_items,                          ///< [in] Total number of items to scan
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        // Scan data type
        typedef typename std::iterator_traits<InputIterator>::value_type T;

        return DeviceScanDispatch<InputIterator, OutputIterator, Sum, T, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            T(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide exclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the exclusive prefix min-scan of an \p int device vector
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // MyMin functor
     * struct MyMin
     * {
     *     template <typename T>
     *     __host__ __device__ __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int      num_items;      // e.g., 7
     * int      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int      *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * MyMin    min_op
     * ...
     *
     * // Determine temporary device storage requirements for exclusive prefix scan
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // Allocate temporary storage for exclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run exclusive prefix min-scan
     * cub::DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, (int) MAX_INT, num_items);
     *
     * // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
     *
     * \endcode
     *
     * \tparam InputIterator    <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator   <b>[inferred]</b> Random-access iterator type for output \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     * \tparam Identity         <b>[inferred]</b> Type of the \p identity value used Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIterator,
        typename        OutputIterator,
        typename        ScanOp,
        typename        Identity>
    __host__ __device__
    static cudaError_t ExclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Iterator pointing to scan input
        OutputIterator  d_out,                              ///< [in] Iterator pointing to scan output
        ScanOp          scan_op,                            ///< [in] Binary scan operator
        Identity        identity,                           ///< [in] Identity element
        int             num_items,                          ///< [in] Total number of items to scan
        cudaStream_t    stream              = 0,            ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous   = false)        ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, ScanOp, Identity, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            identity,
            num_items,
            stream,
            debug_synchronous);
    }


    //@}  end member group
    /******************************************************************//**
     * \name Inclusive scans
     *********************************************************************/
    //@{


    /**
     * \brief Computes a device-wide inclusive prefix sum.
     *
     * \par
     * - Supports non-commutative sum operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the inclusive prefix sum of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int  num_items;      // e.g., 7
     * int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix sum
     * void     *d_temp_storage = NULL;
     * size_t   temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // Allocate temporary storage for inclusive prefix sum
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix sum
     * cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
     *
     * // d_out <-- [8, 14, 21, 26, 29, 29, 38]
     *
     * \endcode
     *
     * \tparam InputIterator      <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator     <b>[inferred]</b> Random-access iterator type for output \iterator
     */
    template <
        typename            InputIterator,
        typename            OutputIterator>
    __host__ __device__
    static cudaError_t InclusiveSum(
        void                *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t              &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator       d_in,                               ///< [in] Iterator pointing to scan input
        OutputIterator      d_out,                              ///< [in] Iterator pointing to scan output
        int                 num_items,                          ///< [in] Total number of items to scan
        cudaStream_t        stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool                debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, Sum, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            Sum(),
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }


    /**
     * \brief Computes a device-wide inclusive prefix scan using the specified binary \p scan_op functor.
     *
     * \par
     * - Supports non-commutative scan operators.
     * - \devicestorage
     * - \cdp
     *
     * \par
     * The code snippet below illustrates the inclusive prefix min-scan of an \p int device vector.
     * \par
     * \code
     * #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
     *
     * // MyMin functor
     * struct MyMin
     * {
     *     template <typename T>
     *     __host__ __device__ __forceinline__
     *     T operator()(const T &a, const T &b) const {
     *         return (b < a) ? b : a;
     *     }
     * };
     *
     * // Declare, allocate, and initialize device pointers for input and output
     * int      num_items;      // e.g., 7
     * int      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
     * int      *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
     * MyMin    min_op;
     * ...
     *
     * // Determine temporary device storage requirements for inclusive prefix scan
     * void *d_temp_storage = NULL;
     * size_t temp_storage_bytes = 0;
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // Allocate temporary storage for inclusive prefix scan
     * cudaMalloc(&d_temp_storage, temp_storage_bytes);
     *
     * // Run inclusive prefix min-scan
     * cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, min_op, num_items);
     *
     * // d_out <-- [8, 6, 6, 5, 3, 0, 0]
     *
     * \endcode
     *
     * \tparam InputIterator    <b>[inferred]</b> Random-access iterator type for input \iterator
     * \tparam OutputIterator   <b>[inferred]</b> Random-access iterator type for output \iterator
     * \tparam ScanOp           <b>[inferred]</b> Binary scan operator type having member <tt>T operator()(const T &a, const T &b)</tt>
     */
    template <
        typename        InputIterator,
        typename        OutputIterator,
        typename        ScanOp>
    __host__ __device__
    static cudaError_t InclusiveScan(
        void            *d_temp_storage,                    ///< [in] %Device allocation of temporary storage.  When NULL, the required allocation size is returned in \p temp_storage_bytes and no work is done.
        size_t          &temp_storage_bytes,                ///< [in,out] Size in bytes of \p d_temp_storage allocation
        InputIterator   d_in,                               ///< [in] Iterator pointing to scan input
        OutputIterator  d_out,                              ///< [in] Iterator pointing to scan output
        ScanOp          scan_op,                            ///< [in] Binary scan operator
        int             num_items,                          ///< [in] Total number of items to scan
        cudaStream_t    stream             = 0,             ///< [in] <b>[optional]</b> CUDA stream to launch kernels within.  Default is stream<sub>0</sub>.
        bool            debug_synchronous  = false)         ///< [in] <b>[optional]</b> Whether or not to synchronize the stream after every kernel launch to check for errors.  May cause significant slowdown.  Default is \p false.
    {
        // Signed integer type for global offsets
        typedef int Offset;

        return DeviceScanDispatch<InputIterator, OutputIterator, ScanOp, NullType, Offset>::Dispatch(
            d_temp_storage,
            temp_storage_bytes,
            d_in,
            d_out,
            scan_op,
            NullType(),
            num_items,
            stream,
            debug_synchronous);
    }

    //@}  end member group

};

/**
 * \example example_device_scan.cu
 */

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)


