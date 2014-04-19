/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#include <stdio.h>

#include "util.h"

// pick the best GPU to run on
void gpu_init(void)
{
    int device_count = 0;
    cudaError_t err;

    cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceLmemResizeToMax);

    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "gpu_init: error enumerating GPUs (%d)\n", err);
        exit(1);
    }

    if (device_count == 0)
    {
        fprintf(stderr, "gpu_init: no devices found\n");
        exit(1);
    }

    // pick the best device to run on
    cudaDeviceProp best_dev_prop;
    int best_dev = 0;
    int dev;

    cudaGetDeviceProperties(&best_dev_prop, 0);

    for(dev = 0; dev < device_count; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        if (prop.major >= best_dev_prop.major &&
            prop.minor >= best_dev_prop.minor)
        {
            best_dev_prop = prop;
            best_dev = dev;
        }
    }

    cudaSetDevice(best_dev);
    fprintf(stderr, "Running on %s (%d MB)\n", best_dev_prop.name, best_dev_prop.totalGlobalMem / 1024 / 1024);
}
