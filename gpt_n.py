import numpy as np
import ctypes
import os
import time
attention_lib = ctypes.cdll.LoadLibrary(os.path.abspath("./shared.so"))

#inference_time = 0.0

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
    total_heads, seq_len, head_dim = q.shape
    q = q.astype(np.float32).flatten()
    k = k.astype(np.float32).flatten()
    v = v.astype(np.float32).flatten()
    mask = mask.astype(np.float32).flatten()
    output = np.zeros_like(q, dtype=np.float32)

    q_ptr = q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_ptr = k.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_ptr = v.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    mask_ptr = mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    attention_lib.attention(
        q_ptr, k_ptr, v_ptr, mask_ptr, output_ptr,
        ctypes.c_int(total_heads), ctypes.c_int(seq_len),
        ctypes.c_int(head_dim)
    )

    output = output.reshape(total_heads, seq_len, head_dim).transpose(1, 0, 2).reshape(seq_len, total_heads*head_dim)
    return output

def mha(x, c_attn, c_proj, n_head):
    seq_len, embed_dim = x.shape
    head_dim = embed_dim // n_head
    q, k, v = np.split(linear(x, **c_attn), 3, axis=-1)
    q = q.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    k = k.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    v = v.reshape(seq_len, n_head, head_dim).transpose(1, 0, 2)
    causal_mask = (1 - np.tri(seq_len, dtype=np.float32)) * -1e10
    attn_output = attention(q, k, v, causal_mask)
    return linear(attn_output, **c_proj)

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    x = x + ffn(layer_norm(x, **ln_2), **mlp)
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    cnt = 0
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return layer_norm(x, **ln_f) @ wte.T

def main(prompt: str, n_tokens_to_generate: int = 20, model_size: str = "124M", models_dir: str = "models", n_iterations: int = 1):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    inputs = input_ids
    n_head = hparams["n_head"]
    inference_times = []

    for i in range(n_iterations):
        start_time = time.time()
        inputs = encoder.encode(prompt)
        for _ in range(n_tokens_to_generate):
            logits = gpt2(inputs, **params, n_head=n_head)
            next_id = np.argmax(logits[-1])
            inputs.append(int(next_id))
            print(encoder.decode([next_id]), end="", flush=True)

        end_time = time.time()
        iteration_time = end_time - start_time
        inference_times.append(iteration_time)
        print(f"\nIteration {i+1}/{n_iterations} completed in {iteration_time:.4f} seconds.")
    print(f"Average inference time: {np.mean(inference_times):.4f} seconds")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
