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

namespace nvbio {

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>::Vec(const T* v)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = v[d];
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>::Vec(const T v)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = v;
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& Vec<DIM,T>::operator= (const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = op2.data[d];

    return *this;
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator+ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] + op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator+= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] + op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator- (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] - op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator-= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] - op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator* (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] - op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator*= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] * op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator/ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] / op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator/= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] / op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> min(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = nvbio::min( op1.data[d], op2.data[d] );
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> max(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = nvbio::max( op1.data[d], op2.data[d] );
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool any(const Vec<DIM,T>& op)
{
    bool r = false;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op.data[d] != 0);
    return r;
}

} // namespace nvbio
