/*
 * Anchor-free DFL decode kernels for the Ultralytics model family
 * (yolov5su / v8 / v11 / v26 heads). Selected at build time via
 * `make HEAD_STYLE=v8` — see docs/ultralytics-support.zh-CN.md.
 *
 * Head output layout per grid position (after the matx transpose):
 *   channels = concat(box-distribution [4*reg_max], cls [nc]), one anchor per
 *   position (anchor-free). reg_max is a runtime parameter (16 for v8/v5su/v11;
 *   1 for yolo26 which drops DFL and regresses the 4 distances directly).
 *
 * Decode math (verified against ultralytics 8.4.142 source):
 *   dfl:       d_side = sum_i softmax(reg_max bins)[i] * i   (nn/modules/block.py DFL)
 *   dist2bbox: x1=(gx+0.5-l)*s, y1=(gy+0.5-t)*s,
 *              x2=(gx+0.5+r)*s, y2=(gy+0.5+b)*s               (utils/tal.py dist2bbox)
 *   score:     sigmoid(cls) — no objectness
 */

#include "decode_nms.h"

#define GPU_BLOCK_THREADS 512

dim3 dfl_grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

dim3 dfl_block_dims(int numJobs) { return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS; }

static __host__ inline float dfl_desigmoid(float y) { return -log(1.0f / y - 1.0f); }

static __device__ inline float dfl_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

static __device__ inline float rd(float v) { return v; }
static __device__ inline float rd(__half v) { return __half2float(v); }

static __device__ inline void dfl_affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1];
    *oy = matrix[0] * y + matrix[2];
}

// DFL integral: softmax-weighted expectation over reg_max bins (raw logits in)
static __device__ inline float dfl_integral(const float *bins, int reg_max)
{
    if (reg_max == 1)
        return bins[0]; // yolo26-style head: the value IS the distance
    float m = bins[0];
    for (int i = 1; i < reg_max; ++i)
        m = fmaxf(m, bins[i]);
    float e_sum = 0.0f, w_sum = 0.0f;
    for (int i = 0; i < reg_max; ++i)
    {
        float e = expf(bins[i] - m);
        e_sum += e;
        w_sum += e * i;
    }
    return w_sum / e_sum;
}

static const int DFL_NUM_BOX_ELEMENT = 7; // left, top, right, bottom, confidence, class, keepflag

// Decode the box (ltrb distances via DFL) for one grid position.
// pitem points at the position's channels; prior at (gx, gy, -, -, stride).
template <typename T>
static __device__ inline void decode_dfl_box(const T *pitem, int reg_max, const float *prior, float *l, float *t,
                                              float *r, float *b)
{
    float  bins[64]; // reg_max <= 16 in practice (4 sides x 16 bins max)
    int    ch        = 0;
    for (int side = 0; side < 4; ++side)
    {
        for (int i = 0; i < reg_max; ++i)
            bins[i] = rd(pitem[ch++]);
        if (side == 0)
            *l = (prior[0] + 0.5f - dfl_integral(bins, reg_max)) * prior[4];
        else if (side == 1)
            *t = (prior[1] + 0.5f - dfl_integral(bins, reg_max)) * prior[4];
        else if (side == 2)
            *r = (prior[0] + 0.5f + dfl_integral(bins, reg_max)) * prior[4];
        else
            *b = (prior[1] + 0.5f + dfl_integral(bins, reg_max)) * prior[4];
    }
}

// --- validation variant: emit every class above threshold (CPU NMS consumes) ---
template <typename T>
static __global__ void decode_dfl_validate_kernel(T *predict, int num_bboxes, int fm_area, int num_classes,
                                                   int reg_max, float confidence_threshold,
                                                   float deconfidence_threshold, float *affine_matrix, float *parray,
                                                   const float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // predict is kGridTotal x (4*reg_max + nc); prior_box is kGridTotal x 5
    const T *pitem = predict + position * (4 * reg_max + num_classes);

    float left, top, right, bottom;
    decode_dfl_box(pitem, reg_max, prior_box + position * 5, &left, &top, &right, &bottom);
    dfl_affine_project(affine_matrix, left, top, &left, &top);
    dfl_affine_project(affine_matrix, right, bottom, &right, &bottom);

    for (int i = 0; i < num_classes; ++i)
    {
        float class_confidence = rd(pitem[4 * reg_max + i]);
        if (class_confidence <= deconfidence_threshold)
            continue;

        class_confidence = dfl_sigmoid(class_confidence);
        if (class_confidence <= confidence_threshold)
            continue;

        int index = atomicAdd(parray, 1);
        if (index >= max_objects)
            return;

        float *pout_item = parray + 1 + index * DFL_NUM_BOX_ELEMENT;
        *pout_item++     = left;
        *pout_item++     = top;
        *pout_item++     = right;
        *pout_item++     = bottom;
        *pout_item++     = class_confidence;
        *pout_item++     = i;
        *pout_item++     = 1; // 1 = keep, 0 = ignore
    }
}

// --- single-image variant: best class only, affine to source coordinates ---
template <typename T>
static __global__ void decode_dfl_kernel(T *predict, int num_bboxes, int fm_area, int num_classes, int reg_max,
                                          float confidence_threshold, float deconfidence_threshold,
                                          float *affine_matrix, float *parray, const float *prior_box, int max_objects)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    const T *pitem = predict + position * (4 * reg_max + num_classes);

    float confidence = rd(pitem[4 * reg_max]);
    int   label      = 0;
    for (int i = 1; i < num_classes; ++i)
    {
        float class_confidence = rd(pitem[4 * reg_max + i]);
        if (class_confidence > confidence)
        {
            confidence = class_confidence;
            label      = i;
        }
    }

    confidence = dfl_sigmoid(confidence);
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;

    float left, top, right, bottom;
    decode_dfl_box(pitem, reg_max, prior_box + position * 5, &left, &top, &right, &bottom);
    dfl_affine_project(affine_matrix, left, top, &left, &top);
    dfl_affine_project(affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + 1 + index * DFL_NUM_BOX_ELEMENT;
    *pout_item++     = left;
    *pout_item++     = top;
    *pout_item++     = right;
    *pout_item++     = bottom;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
}

void decode_dfl_validate_kernel_invoker(float *predict, int num_bboxes, int fm_area, int num_classes, int reg_max,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = dfl_grid_dims(num_bboxes);
    auto block = dfl_block_dims(num_bboxes);
    checkCudaKernel(decode_dfl_validate_kernel<float><<<grid, block, 0, stream>>>(
        predict, num_bboxes, fm_area, num_classes, reg_max, confidence_threshold,
        dfl_desigmoid(confidence_threshold), affine_matrix, parray, prior_box, max_objects));
}

void decode_dfl_validate_kernel_invoker(half *predict, int num_bboxes, int fm_area, int num_classes, int reg_max,
                                        float confidence_threshold, float nms_threshold, float *affine_matrix,
                                        float *parray, const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = dfl_grid_dims(num_bboxes);
    auto block = dfl_block_dims(num_bboxes);
    checkCudaKernel(decode_dfl_validate_kernel<half><<<grid, block, 0, stream>>>(
        (half *)predict, num_bboxes, fm_area, num_classes, reg_max, confidence_threshold,
        dfl_desigmoid(confidence_threshold), affine_matrix, parray, prior_box, max_objects));
}

void decode_dfl_kernel_invoker(float *predict, int num_bboxes, int fm_area, int num_classes, int reg_max,
                               float confidence_threshold, float nms_threshold, float *affine_matrix, float *parray,
                               const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = dfl_grid_dims(num_bboxes);
    auto block = dfl_block_dims(num_bboxes);
    checkCudaKernel(decode_dfl_kernel<float><<<grid, block, 0, stream>>>(
        predict, num_bboxes, fm_area, num_classes, reg_max, confidence_threshold,
        dfl_desigmoid(confidence_threshold), affine_matrix, parray, prior_box, max_objects));
}

void decode_dfl_kernel_invoker(half *predict, int num_bboxes, int fm_area, int num_classes, int reg_max,
                               float confidence_threshold, float nms_threshold, float *affine_matrix, float *parray,
                               const float *prior_box, int max_objects, cudaStream_t stream)
{
    auto grid  = dfl_grid_dims(num_bboxes);
    auto block = dfl_block_dims(num_bboxes);
    checkCudaKernel(decode_dfl_kernel<half><<<grid, block, 0, stream>>>(
        (half *)predict, num_bboxes, fm_area, num_classes, reg_max, confidence_threshold,
        dfl_desigmoid(confidence_threshold), affine_matrix, parray, prior_box, max_objects));
}
