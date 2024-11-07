#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
// fork from https://github.com/130bb56/Graduation-Project/blob/main/shared/shared.cu
extern "C" {

__global__ void attention_kernel(
    const __half* __restrict__ q,    // (total_heads, seq_len, head_dim)
    const __half* __restrict__ k,    // (total_heads, seq_len, head_dim)
    const __half* __restrict__ v,    // (total_heads, seq_len, head_dim)
    const __half* __restrict__ mask, // (seq_len, seq_len)
    __half* __restrict__ output,     // (total_heads, seq_len, head_dim)
    /*
    const __half *q,    // (total_heads, seq_len, head_dim)
    const __half *k,    // (total_heads, seq_len, head_dim)
    const __half *v,    // (total_heads, seq_len, head_dim)
    const __half *mask, // (seq_len, seq_len)
    __half *output,     // (total_heads, seq_len, head_dim)
    */
    int total_heads,
    int seq_len,
    int head_dim) {

    extern __shared__ __half sram[]; // Load shared mem
    int head_idx = blockIdx.x; // total_heads
    int q_pos = threadIdx.x;   // seq_len

    if (head_idx >= total_heads || q_pos >= seq_len)
        return;

    const __half *q_ptr = q + head_idx * seq_len * head_dim + q_pos * head_dim;
    const __half *k_ptr = k + head_idx * seq_len * head_dim;
    const __half *v_ptr = v + head_idx * seq_len * head_dim;
    __half *output_ptr = output + head_idx * seq_len * head_dim + q_pos * head_dim;

    __half *q_ = sram;
    __half *k_ = q_ + seq_len * head_dim;
    __half *v_ = k_ + seq_len * head_dim;
    __half *mask_ = v_ + seq_len * head_dim;
    __half *scores = mask_+ seq_len * seq_len;

    const __half scale = __float2half(1.0f / sqrtf((float)head_dim));

    for (int d = 0; d < head_dim; ++d) {
        q_[q_pos * head_dim + d] = q_ptr[d];
        k_[q_pos * head_dim + d] = k_ptr[q_pos * head_dim + d];
        v_[q_pos * head_dim + d] = v_ptr[q_pos * head_dim + d];
    }
    for (int k_pos = 0; k_pos < seq_len; k_pos++) {
        mask_[q_pos * seq_len + k_pos] = mask[q_pos * seq_len + k_pos];
        scores[q_pos * seq_len + k_pos] = __float2half(0.0f);
    }
    __syncthreads();
    __half max_score = __float2half(-1e4f);

    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        //scores[k_pos] = __float2half(0.0f);
        int idx = seq_len * q_pos + k_pos;
        for (int d = 0; d < head_dim; ++d) {
            scores[idx] = __hadd(scores[idx], __hmul(q_[q_pos * head_dim + d], k_[k_pos * head_dim + d]));
        }
        scores[idx] = __hadd(__hmul(scores[idx], scale), mask_[q_pos * seq_len + k_pos]);
        if (__hgt(scores[idx], max_score)) {
            max_score = scores[idx];
        }
    }
    __half sum_exp = __float2half(0.0f);
    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        scores[seq_len * q_pos + k_pos] = expf(scores[seq_len * q_pos + k_pos] - max_score);
        sum_exp = __hadd(sum_exp, scores[seq_len * q_pos + k_pos]);
    }

    for (int d = 0; d < head_dim; ++d) {
        __half result = __float2half(0.0f);
        for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
            __half attn_weight = __hdiv(scores[seq_len * q_pos + k_pos], sum_exp);
            //float v_val = v_ptr[k_pos * head_dim + d];
            result = __hadd(result, __hmul(attn_weight, v_[k_pos * head_dim + d]));
        }
        output_ptr[d] = result;
    }

}

void attention(
    const __half *q,
    const __half *k,
    const __half *v,
    const __half *mask,
    __half *output,
    int total_heads,
    int seq_len,
    int head_dim) {

    __half *d_q, *d_k, *d_v, *d_mask, *d_output;
    cudaStream_t stream;
    cudaError_t cuda_status;

    cuda_status = cudaStreamCreate(&stream);
    if (cuda_status != cudaSuccess) {
        printf("Stream creation failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    size_t size_qkv = sizeof(__half) * total_heads * seq_len * head_dim;
    size_t size_mask = sizeof(__half) * seq_len * seq_len;

    cudaMalloc((void**)&d_q, size_qkv);
    cudaMalloc((void**)&d_k, size_qkv);
    cudaMalloc((void**)&d_v, size_qkv);
    cudaMalloc((void**)&d_mask, size_mask);
    cudaMalloc((void**)&d_output, size_qkv);
    cudaMemcpyAsync(d_q, q, size_qkv, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_k, k, size_qkv, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v, v, size_qkv, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_mask, mask, size_mask, cudaMemcpyHostToDevice, stream);

    int threads = seq_len;
    int blocks = total_heads;
    int shared_mem_size = sizeof(__half) * seq_len * (head_dim * 3 + seq_len * 2);

    attention_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_q, d_k, d_v, d_mask, d_output,
        total_heads, seq_len, head_dim
    );

    cudaGetLastError();
    cudaMemcpyAsync(output, d_output, size_qkv, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_mask);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
}

}
