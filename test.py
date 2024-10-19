import numpy as np
import ctypes
import os

# CUDA 라이브러리 로드
attention_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./test.so"))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    # q, k, v의 형태: (batch_size * n_head, seq_len, head_dim)
    total_heads, seq_len, head_dim = q.shape

    q_flat = q.astype(np.float32).flatten()
    k_flat = k.astype(np.float32).flatten()
    v_flat = v.astype(np.float32).flatten()
    mask_flat = mask.astype(np.float32).flatten()
    output_flat = np.zeros_like(q_flat, dtype=np.float32)

    q_ptr = q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_ptr = v_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mask_ptr = mask_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # CUDA 함수의 인자 타입을 설정합니다.
    attention_lib.attention.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # q
        ctypes.POINTER(ctypes.c_float),  # k
        ctypes.POINTER(ctypes.c_float),  # v
        ctypes.POINTER(ctypes.c_float),  # mask
        ctypes.POINTER(ctypes.c_float),  # output
        ctypes.c_int,  # total_heads
        ctypes.c_int,  # seq_len
        ctypes.c_int   # head_dim
    ]

    # CUDA 함수 호출
    attention_lib.attention(
        q_ptr, k_ptr, v_ptr, mask_ptr, output_ptr,
        ctypes.c_int(total_heads), ctypes.c_int(seq_len),
        ctypes.c_int(head_dim)
    )

    output = output_flat.reshape(total_heads, seq_len, head_dim)
    return output

def mha(x, c_attn, c_proj, n_head):
    batch_size, seq_len, embed_dim = x.shape
    batch_size = 1;
    head_dim = embed_dim // n_head

    x = linear(x, **c_attn)
    # q, k, v 분리 및 헤드로 분할
    qkv = np.split(x, 3, axis=-1)
    q = qkv[0].reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    k = qkv[1].reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)
    v = qkv[2].reshape(batch_size, seq_len, n_head, head_dim).transpose(0, 2, 1, 3)

    # CUDA 함수는 (batch_size * n_head, seq_len, head_dim) 형태의 q, k, v를 받습니다.
    q = q.reshape(-1, seq_len, head_dim)
    k = k.reshape(-1, seq_len, head_dim)
    v = v.reshape(-1, seq_len, head_dim)

    # Causal mask 생성
    causal_mask = (1 - np.tri(seq_len, dtype=np.float32)) * -1e10

    # 어텐션 계산
    attn_output = attention(q, k, v, causal_mask)

    # 헤드 결합
    attn_output = attn_output.reshape(batch_size, n_head, seq_len, head_dim).transpose(0, 2, 1, 3)
    attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

    x = linear(attn_output, **c_proj)
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    x = x[np.newaxis, ...]  # 배치 차원 추가
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)
    logits = (x @ wte.T)[0]  # 배치 차원 제거
    return logits

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, **params, n_head=n_head)
        next_id = np.argmax(logits[-1])
        inputs.append(int(next_id))
    return inputs[len(inputs) - n_tokens_to_generate:]

def main(prompt: str, n_tokens_to_generate: int = 20, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    print(output_text)

if __name__ == "__main__":
    import fire
    fire.Fire(main)

