#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

extern "C" {

__global__ void attention_kernel(
    const float *q,    // (total_heads, seq_len, head_dim)
    const float *k,    // (total_heads, seq_len, head_dim)
    const float *v,    // (total_heads, seq_len, head_dim)
    const float *mask, // (seq_len, seq_len)
    float *output,     // (total_heads, seq_len, head_dim)
    int total_heads,
    int seq_len,
    int head_dim) {

    extern __shared__ float sram[]; // Load shared mem
    int head_idx = blockIdx.x; // total_heads
    int q_pos = threadIdx.x;   // seq_len

    if (head_idx >= total_heads || q_pos >= seq_len)
        return;

    const float *q_ptr = q + head_idx * seq_len * head_dim + q_pos * head_dim;
    const float *k_ptr = k + head_idx * seq_len * head_dim;
    const float *v_ptr = v + head_idx * seq_len * head_dim;
    float *output_ptr = output + head_idx * seq_len * head_dim + q_pos * head_dim;
    
    float *q_ = sram;
    float *k_ = q_ + seq_len * head_dim;
    float *v_ = k_ + seq_len * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int d = 0; d < head_dim; ++d) {
        q_[q_pos * head_dim + d] = q_ptr[d];
        k_[q_pos * head_dim + d] = k_ptr[q_pos * head_dim + d];
        v_[q_pos * head_dim + d] = v_ptr[q_pos * head_dim + d];
    }
    __syncthreads();

    float max_score = -1e5f;
    float scores[40];

    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        scores[k_pos] = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            scores[k_pos] += q_[q_pos * head_dim + d] * k_[k_pos * head_dim + d];
        }
        scores[k_pos] = scores[k_pos] * scale + mask[q_pos * seq_len + k_pos];
        scores[k_pos] += mask[q_pos * seq_len + k_pos];
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    float sum_exp = 0.0f;
    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        scores[k_pos] = expf(scores[k_pos] - max_score);
        sum_exp += scores[k_pos];
    }

    for (int d = 0; d < head_dim; ++d) {
        float result = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
            float attn_weight = scores[k_pos] / sum_exp;
            result += attn_weight * v_[k_pos * head_dim + d];
        }
        output_ptr[d] = result;
    }
}

void attention(
    const float *q,    // (total_heads, seq_len, head_dim)
    const float *k,    // (total_heads, seq_len, head_dim)
    const float *v,    // (total_heads, seq_len, head_dim)
    const float *mask, // (seq_len, seq_len)
    float *output,     // (total_heads, seq_len, head_dim)
    int total_heads,
    int seq_len,
    int head_dim) {

    float *d_q, *d_k, *d_v, *d_mask, *d_output;

    size_t size_qkv = sizeof(float) * total_heads * seq_len * head_dim;
    size_t size_mask = sizeof(float) * seq_len * seq_len;

    cudaMalloc((void**)&d_q, size_qkv);
    cudaMalloc((void**)&d_k, size_qkv);
    cudaMalloc((void**)&d_v, size_qkv);
    cudaMalloc((void**)&d_mask, size_mask);
    cudaMalloc((void**)&d_output, size_qkv);

    cudaMemcpy(d_q, q, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    int threads = seq_len;
    int blocks = total_heads;
    int shared_mem_size = sizeof(float) * seq_len * head_dim * 3;
    //printf("%d\n", shared_mem_size);
    attention_kernel<<<blocks, threads, shared_mem_size>>>(
        d_q, d_k, d_v, d_mask, d_output,
        total_heads, seq_len, head_dim
    );

    cudaMemcpy(output, d_output, size_qkv, cudaMemcpyDeviceToHost);
    //printf("%lu\n", size_qkv);

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_mask);
    cudaFree(d_output);
}

} 
