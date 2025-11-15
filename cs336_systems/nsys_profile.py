import torch
import torch.cuda.nvtx as nvtx
from torch import Tensor
import numpy as np
import timeit
import cs336_systems
from cs336_systems.model import BasicsTransformerLM
from einops import rearrange, einsum
from jaxtyping import Float, Bool, Int
from nn_utils import softmax
import math

def run_basics_transformer_model(size, d_model, d_ff, num_layers, num_heads, w_num_steps, num_steps):
    print(f"=================Benchmark for model {size} started=================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=128,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=0.5).to(device)
    
    
    
    x = torch.randint(1, 1000, (1, 100), device=device)

    for step in range(w_num_steps):
        print(f"\rWarm-up step forward pass: {step}", end="")
        y = model(x).mean()

    forward_time = []
    backward_time = []
    for step in range(num_steps):
        print(f"\rBenchmark step forward pass: {step}", end="")
        t1 = timeit.default_timer()
        y = model(x).mean()
        torch.cuda.synchronize()
        t2 = timeit.default_timer()
        forward_time.append(t2-t1)

        print(f"\rBenchmark step backward pass: {step}", end="")
        t1 = timeit.default_timer()
        y.backward()
        torch.cuda.synchronize()
        t2 = timeit.default_timer()
        backward_time.append(t2-t1)
        
    print(f"=================Benchmark for model {size} finished=================")
    print(f"Forward pass timing average: {np.average(forward_time)}")
    print(f"Forward pass timing stadard deviation: {np.std(np.array(forward_time))}")
    print(f"Backward pass timing average: {np.average(backward_time)}")
    print(f"Backward pass timing stadard deviation: {np.std(np.array(backward_time))}")

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    with nvtx.range("computing attention scores"):
        d_k = K.shape[-1]
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        matmul = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    
    return matmul
    

if __name__ == "__main__":
    cs336_systems.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    run_basics_transformer_model(size="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10)

    #run_basics_transformer_model(size="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10)

    #run_basics_transformer_model(size="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10)

    #run_basics_transformer_model(size="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10)

    #run_basics_transformer_model(size="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10)