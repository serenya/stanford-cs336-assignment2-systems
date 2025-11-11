import torch
import numpy as np
import timeit
from cs336_systems.model import BasicsTransformerLM

def run_basics_transformer_model(size, d_model, d_ff, num_layers, num_heads, w_num_steps, num_steps):
    print(f"=================Benchmark for model {size} started=================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=100000,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=0.5).to(device)
    
    
    
    x = torch.randint(1, 1000, (1, 99000), device=device)

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
    

if __name__ == "__main__":
    run_basics_transformer_model(size="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10)

    run_basics_transformer_model(size="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10)

    run_basics_transformer_model(size="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10)

    run_basics_transformer_model(size="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10)

    run_basics_transformer_model(size="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10)