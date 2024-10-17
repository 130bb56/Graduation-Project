import numpy as np
import ctypes
import os

# CUDA 커널 로드
# 현재 디렉토리에서 attention.so를 찾습니다.
attention_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./attention.so"))

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    # 입력 배열을 플래튼하고 float32로 변환
    seq_len, depth = q.shape
    q_flat = q.astype(np.float32).flatten()
    k_flat = k.astype(np.float32).flatten()
    v_flat = v.astype(np.float32).flatten()
    mask_flat = mask.astype(np.float32).flatten()
    output_flat = np.zeros_like(q_flat, dtype=np.float32)

    # 포인터 생성
    q_ptr = q_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_ptr = v_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mask_ptr = mask_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # CUDA 함수 설정
    attention_lib.attention.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int
    ]

    # CUDA 함수 호출
    attention_lib.attention(
        q_ptr, k_ptr, v_ptr, mask_ptr, output_ptr, ctypes.c_int(seq_len), ctypes.c_int(depth)
    )

    # 결과를 원래 형태로 변환
    output = output_flat.reshape(seq_len, depth)
    return output

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, c_attn['w'], c_attn['b'])
    qkv = np.split(x, 3, axis=-1)
    q_heads = np.split(qkv[0], n_head, axis=-1)
    k_heads = np.split(qkv[1], n_head, axis=-1)
    v_heads = np.split(qkv[2], n_head, axis=-1)

    seq_len = x.shape[0]
    causal_mask = (1 - np.tri(seq_len, k=0, dtype=np.float32)) * -1e10

    out_heads = []
    for q, k, v in zip(q_heads, k_heads, v_heads):
        out = attention(q, k, v, causal_mask)
        out_heads.append(out)

    x = np.concatenate(out_heads, axis=-1)
    x = linear(x, c_proj['w'], c_proj['b'])
    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

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
    #print(output_text)
    return output_text

if __name__ == "__main__":
    import fire
    fire.Fire(main)

