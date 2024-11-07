#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void compute_attention(
    const float *q, const float *k, const float *mask,
    float *scores, int seq_len, int depth) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // Query index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Key index

    if (i < seq_len && j < seq_len) {
        float score = 0.0f;
        for (int d = 0; d < depth; ++d) {
            score += q[i * depth + d] * k[j * depth + d];
        }
        score /= sqrtf((float)depth);
        score += mask[i * seq_len + j];
        scores[i * seq_len + j] = score;
    }
}

__global__ void softmax_kernel(float *scores, int seq_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len) {
        float max_score = -1e20f;
        for (int j = 0; j < seq_len; ++j) {
            float val = scores[i * seq_len + j];
            if (val > max_score) max_score = val;
        }
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            float val = expf(scores[i * seq_len + j] - max_score);
            scores[i * seq_len + j] = val;
            sum_exp += val;
        }
        for (int j = 0; j < seq_len; ++j) {
            scores[i * seq_len + j] /= sum_exp;
        }
    }
}

__global__ void compute_output(
    const float *scores, const float *v, float *output,
    int seq_len, int depth) {

    int i = blockIdx.x * blockDim.x + threadIdx.x; // Query index
    int d = blockIdx.y * blockDim.y + threadIdx.y; // Depth index

    if (i < seq_len && d < depth) {
        float result = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            result += scores[i * seq_len + j] * v[j * depth + d];
        }
        output[i * depth + d] = result;
    }
}

void attention(
    const float *q, const float *k, const float *v, const float *mask,
    float *output, int seq_len, int depth) {

    float *d_q, *d_k, *d_v, *d_mask, *d_scores, *d_output;

    size_t size_qkv = sizeof(float) * seq_len * depth;
    size_t size_scores = sizeof(float) * seq_len * seq_len;

    cudaMalloc((void**)&d_q, size_qkv);
    cudaMalloc((void**)&d_k, size_qkv);
    cudaMalloc((void**)&d_v, size_qkv);
    cudaMalloc((void**)&d_mask, size_scores);
    cudaMalloc((void**)&d_scores, size_scores);
    cudaMalloc((void**)&d_output, size_qkv);

    cudaMemcpy(d_q, q, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_scores, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((seq_len + blockDim.x - 1) / blockDim.x, (seq_len + blockDim.y - 1) / blockDim.y);

    compute_attention<<<gridDim, blockDim>>>(d_q, d_k, d_mask, d_scores, seq_len, depth);

    int threads = 32;
    int blocks = (seq_len + threads - 1) / threads;

    softmax_kernel<<<blocks, threads>>>(d_scores, seq_len);

    dim3 blockDim2(32, 32);
    dim3 gridDim2((seq_len + blockDim2.x - 1) / blockDim2.x, (depth + blockDim2.y - 1) / blockDim2.y);

    compute_output<<<gridDim2, blockDim2>>>(d_scores, d_v, d_output, seq_len, depth);

    // Copy result back to host
    cudaMemcpy(output, d_output, size_qkv, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_mask);
    cudaFree(d_scores);
    cudaFree(d_output);
}

}
