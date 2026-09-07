/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "matx_reformat.h"
#include "matx.h"

#define CHECK_CUDA(call) \
do { \
    cudaError_t status = call; \
    if (status != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        exit(1); \
    } \
} while(0)

// ----------------------- Half to float -----------------------
__global__ void convert_half_to_float_kernel(const __half* a, float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert half to float and store in b
        b[idx] = __half2float(a[idx]);
    }
}

void convert_half_to_float(__half* a, float* b, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_half_to_float_kernel<<<num_blocks, threads_per_block>>>(a, b, size);
}

// ----------------------- Float to half -----------------------
__global__ void convert_float_to_half_kernel(const float* a, __half* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Convert half to float and store in b
        b[idx] = __float2half(a[idx]);
    }
}

void convert_float_to_half(float* a, __half* b, int size) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_float_to_half_kernel<<<num_blocks, threads_per_block>>>(a, b, size);
}

// ----------------------- int8 to half -----------------------
__global__ void convert_int8_to_half_kernel(const int8_t* a, __half* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = static_cast<__half>(a[idx]) * __float2half(scale);
    }
}

void convert_int8_to_half(int8_t* a, __half* b, int size, float scale) {
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_int8_to_half_kernel<<<num_blocks, threads_per_block>>>(a, b, size, scale);
}

// ----------------------- Float to int8 -----------------------
__global__ void convert_float_to_int8_kernel(const float* a, int8_t* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = (a[idx] / scale);
        if(v < -128) v = -128;
        if(v > 127) v = 127;
        b[idx] = (int8_t)v;
    }
}
void convert_float_to_int8(float* a, int8_t* b, int size, float scale){
    const int threads_per_block = 256;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_float_to_int8_kernel<<<num_blocks, threads_per_block>>>(a, b, size, scale);
}

void* allocDeviceMemory(size_t size)
{
    void* device_ptr;
    CHECK_CUDA(cudaMalloc(&device_ptr, size));
    return device_ptr;
}

// Channel geometry of the YOLOv5 heads — num classes comes from the app build
// (`NUM_CLASSES=<n>`, default 80 = COCO model; 3 = the custom 3-class model);
// keep it in sync with the loadable passed to --engine.
// channels per head = 3 anchors * (num_classes + 5); DLA CHW16 pads channels
// to a multiple of 16, CHW32 (V2 paths) to a multiple of 32.
#ifndef YOLO_NUM_CLASSES
#define YOLO_NUM_CLASSES 80
#endif
constexpr int kNumClasses  = YOLO_NUM_CLASSES;

// Head style (keep in sync with the app build): v5 = anchor-based
// (3 anchors x (nc+5) channels); v8 = anchor-free DFL (4*reg_max + nc per
// position, single box per grid cell). Selected via build HEAD_STYLE=v8.
#ifdef YOLO_HEAD_STYLE_V8
#ifndef YOLO_REG_MAX
#define YOLO_REG_MAX 16
#endif
constexpr int kRegMax        = YOLO_REG_MAX;
constexpr int kAnchorsPerPos = 1;
constexpr int kChPerPos      = 4 * kRegMax + kNumClasses;
#else
constexpr int kAnchorsPerPos = 3;
constexpr int kChPerPos      = kNumClasses + 5;
#endif
constexpr int kChPerAnchor = kChPerPos;                      // channels per anchor/position
constexpr int kHeadCh      = kAnchorsPerPos * kChPerAnchor;
constexpr int kChw16Groups = (kHeadCh + 15) / 16;           // 2  (24 -> padded 32)
constexpr int kChw32Groups = (kHeadCh + 31) / 32;           // 1  (24 -> padded 32)


// Network input resolution — must match the app build (`make INPUT_W=<> INPUT_H=<>`,
// both multiples of 32); defaults reproduce the original 672x672 sample.
// Head grid dims / areas derive from it (values shown for 672x672).
#ifndef YOLO_INPUT_W
#define YOLO_INPUT_W 672
#endif
#ifndef YOLO_INPUT_H
#define YOLO_INPUT_H 672
#endif
constexpr int kInW    = YOLO_INPUT_W;
constexpr int kInH    = YOLO_INPUT_H;
constexpr int kW8     = kInW / 8, kH8 = kInH / 8;      // 84x84  @ 672x672
constexpr int kW16    = kInW / 16, kH16 = kInH / 16;   // 42x42
constexpr int kW32    = kInW / 32, kH32 = kInH / 32;   // 21x21
constexpr int kArea8  = kW8 * kH8;                     // 7056
constexpr int kArea16 = kW16 * kH16;                   // 1764
constexpr int kArea32 = kW32 * kH32;                   // 441

class ReformatRunner::ReformatRunnerImpl {
public:
    ~ReformatRunnerImpl()
    {
        CHECK_CUDA(cudaFree(mTemp));
    }

    void Init()
    {
        mTemp = allocDeviceMemory(sizeof(half) * kChw16Groups * 16 * kArea8);
    }

    bool Run(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, kChw16Groups, kH8, kW8, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw16Groups, 16, kH8, kW8});
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea8});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, kArea8, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, kChw16Groups, kH16, kW16, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw16Groups, 16, kH16, kW16});
        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea16});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, kArea16, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, kChw16Groups, kH32, kW32, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw16Groups, 16, kH32, kW32});
        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea32});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, kArea32, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool ReformatImage(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mInput1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, 16, kInH, kInW, 1});
        matx::tensor_t<matx::matxFp16, 5> mOutput1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, 1, kInH, kInW, 16});
        matx::copy(mOutput1, mInput1.Permute({0, 4, 2, 3, 1}), stream);
        return true;
    }

    bool Reformat(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, kChw16Groups, kH8, kW8, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, kChw16Groups, 16, kH8, kW8});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, kChw16Groups, kH16, kW16, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, kChw16Groups, 16, kH16, kW16});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, kChw16Groups, kH32, kW32, 16});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, kChw16Groups, 16, kH32, kW32});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);

        return true;
    }

    bool Transpose(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, kAnchorsPerPos, kChPerAnchor, kArea8});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, kArea8, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, kAnchorsPerPos, kChPerAnchor, kArea16});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, kArea16, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, kAnchorsPerPos, kChPerAnchor, kArea32});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, kArea32, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool RunV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, kChw32Groups, kH8, kW8, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw32Groups, 32, kH8, kW8});
        matx::tensor_t<matx::matxFp16, 4> mHead1Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea8});
        matx::tensor_t<matx::matxFp16, 4> mHead1Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, kArea8, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead1Output2, mHead1Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, kChw32Groups, kH16, kW16, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw32Groups, 32, kH16, kW16});
        matx::tensor_t<matx::matxFp16, 4> mHead2Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea16});
        matx::tensor_t<matx::matxFp16, 4> mHead2Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, kArea16, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead2Output2, mHead2Input2.Permute({0, 3, 1, 2}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, kChw32Groups, kH32, kW32, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kChw32Groups, 32, kH32, kW32});
        matx::tensor_t<matx::matxFp16, 4> mHead3Input2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)mTemp, {1, kAnchorsPerPos, kChPerAnchor, kArea32});
        matx::tensor_t<matx::matxFp16, 4> mHead3Output2 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, kArea32, kAnchorsPerPos, kChPerAnchor});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);
        matx::copy(mHead3Output2, mHead3Input2.Permute({0, 3, 1, 2}), stream);
        return true;
    }

    bool ReformatImageV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<float, 5> mInput1 = matx::make_tensor<float>((float*)src[0], {1, 4, kInH, kInW, 1});
        matx::tensor_t<float, 5> mOutput1 = matx::make_tensor<float>((float*)dst[0], {1, 1, kInH, kInW, 4});
        matx::copy(mOutput1, mInput1.Permute({0, 4, 2, 3, 1}), stream);
        return true;
    }

    bool ReformatV2(void** src, void** dst, cudaStream_t stream)
    {
        matx::tensor_t<matx::matxFp16, 5> mHead1Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[0], {1, kChw32Groups, kH8, kW8, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead1Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[0], {1, kChw32Groups, 32, kH8, kW8});
        matx::copy(mHead1Output1, mHead1Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead2Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[1], {1, kChw32Groups, kH16, kW16, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead2Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[1], {1, kChw32Groups, 32, kH16, kW16});
        matx::copy(mHead2Output1, mHead2Input1.Permute({0, 1, 4, 2, 3}), stream);

        matx::tensor_t<matx::matxFp16, 5> mHead3Input1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)src[2], {1, kChw32Groups, kH32, kW32, 32});
        matx::tensor_t<matx::matxFp16, 5> mHead3Output1 = matx::make_tensor<matx::matxFp16>((matx::matxFp16*)dst[2], {1, kChw32Groups, 32, kH32, kW32});
        matx::copy(mHead3Output1, mHead3Input1.Permute({0, 1, 4, 2, 3}), stream);

        return true;
    }

private:
    void* mTemp;
};

ReformatRunner::ReformatRunner()
{
    pImpl = new ReformatRunnerImpl();
    pImpl->Init();
}

ReformatRunner::~ReformatRunner()
{
    delete pImpl;
}

bool ReformatRunner::Run(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Run(src, dst, stream);
}

bool ReformatRunner::ReformatImage(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatImage(src, dst, stream);
}

bool ReformatRunner::Reformat(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Reformat(src, dst, stream);
}

bool ReformatRunner::Transpose(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->Transpose(src, dst, stream);
}

bool ReformatRunner::RunV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->RunV2(src, dst, stream);
}

bool ReformatRunner::ReformatImageV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatImageV2(src, dst, stream);
}

bool ReformatRunner::ReformatV2(void** src, void** dst, cudaStream_t stream)
{
    return pImpl->ReformatV2(src, dst, stream);
}