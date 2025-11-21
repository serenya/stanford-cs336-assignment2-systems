import torch
import numpy as np
import timeit

from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import pandas as pd

def run_basics_transformer_model(size, d_model, d_ff, num_layers, num_heads, w_num_steps, num_steps, use_autocast):
    print(f"=================Benchmark for model {size} {"mixed precision" if use_autocast else "full precision"} started=================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=256,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=0.5).to(device)
    
    x = torch.randint(0, 10000, (4, 256), device=device)
    y = torch.randint(0, 10000, (4, 256), device=device)

    """ for step in range(w_num_steps):
        print(f"\rWarm-up step forward pass: {step}", end="")
        logits = model(x) """

    forward_time = []
    backward_time = []
    autocast_context = torch.autocast(device_type='cuda') if use_autocast else nullcontext()
    for step in range(num_steps):
        print(f"\rBenchmark step forward pass: {step}", end="")
        t1 = timeit.default_timer()
        with autocast_context:
            logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t2 = timeit.default_timer()
            forward_time.append(t2-t1)

            print(f"\rBenchmark step backward pass: {step}", end="")
            t1 = timeit.default_timer()
            loss = cross_entropy(logits, y)
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = timeit.default_timer()
        backward_time.append(t2-t1)
    

    print(f"=================Benchmark for model {size} {"mixed precision" if use_autocast else "full precision"} finished=================")
    forward_time_average = np.average(forward_time)
    forward_time_standard_deviation = np.std(np.array(forward_time))
    backward_time_average = np.average(backward_time)
    backward_time_standard_deviation = np.std(np.array(backward_time))

    print(f"Forward pass timing average: {forward_time_average}")
    print(f"Forward pass timing stadard deviation: {forward_time_standard_deviation}")
    print(f"Backward pass timing average: {backward_time_average}")
    print(f"Backward pass timing stadard deviation: {backward_time_standard_deviation}")

    return (size, forward_time_average, forward_time_standard_deviation, backward_time_average, backward_time_standard_deviation)

if __name__ == "__main__":
    results = []

    results.append(run_basics_transformer_model(size="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="small", d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10, use_autocast=True))

    df = pd.DataFrame(results, columns=['Model', 'Forward Time Avg', 'Forward Time Std', 'Backward Time Avg', 'Backward Time Std', 'Full precision'])

    print("\n=================Benchmark Results=================")
    print(df.to_markdown(index=False))
    print("===================================================\n")