// attention.cu
#include <cuda_runtime.h>
#include <math.h>

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

    int head_idx = blockIdx.x; // total_heads
    int q_pos = threadIdx.x;   // seq_len

    if (head_idx >= total_heads || q_pos >= seq_len)
        return;

    const float *q_ptr = q + head_idx * seq_len * head_dim + q_pos * head_dim;
    const float *k_ptr = k + head_idx * seq_len * head_dim;
    const float *v_ptr = v + head_idx * seq_len * head_dim;
    float *output_ptr = output + head_idx * seq_len * head_dim + q_pos * head_dim;

    // 어텐션 스코어 계산
    float max_score = -1e20f;
    float *scores = new float[seq_len];

    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            float q_val = q_ptr[d];
            float k_val = k_ptr[k_pos * head_dim + d];
            score += q_val * k_val;
        }
        score /= sqrtf((float)head_dim);
        score += mask[q_pos * seq_len + k_pos];  // 마스크 적용
        scores[k_pos] = score;
        if (score > max_score)
            max_score = score;
    }

    // 소프트맥스 계산을 위한 지수 함수 및 합계 계산
    float sum_exp = 0.0f;
    for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
        float exp_score = expf(scores[k_pos] - max_score);
        scores[k_pos] = exp_score;
        sum_exp += exp_score;
    }

    // 최종 출력 계산
    for (int d = 0; d < head_dim; ++d) {
        float result = 0.0f;
        for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
            float attn_weight = scores[k_pos] / sum_exp;
            float v_val = v_ptr[k_pos * head_dim + d];
            result += attn_weight * v_val;
        }
        output_ptr[d] = result;
    }

    delete[] scores;
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

    // 디바이스 메모리 포인터
    float *d_q, *d_k, *d_v, *d_mask, *d_output;

    size_t size_qkv = sizeof(float) * total_heads * seq_len * head_dim;
    size_t size_mask = sizeof(float) * seq_len * seq_len;

    // 디바이스 메모리 할당
    cudaMalloc((void**)&d_q, size_qkv);
    cudaMalloc((void**)&d_k, size_qkv);
    cudaMalloc((void**)&d_v, size_qkv);
    cudaMalloc((void**)&d_mask, size_mask);
    cudaMalloc((void**)&d_output, size_qkv);

    // 호스트에서 디바이스로 데이터 복사
    cudaMemcpy(d_q, q, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, size_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, size_mask, cudaMemcpyHostToDevice);

    // 커널 런치 구성
    int threads = seq_len;
    int blocks = total_heads;

    // 어텐션 커널 실행
    attention_kernel<<<blocks, threads>>>(
        d_q, d_k, d_v, d_mask, d_output,
        total_heads, seq_len, head_dim);

    // 결과를 호스트로 복사
    cudaMemcpy(output, d_output, size_qkv, cudaMemcpyDeviceToHost);

    // 디바이스 메모리 해제
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_mask);
    cudaFree(d_output);
}

} // extern "C"

