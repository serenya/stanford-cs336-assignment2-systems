import torch
import numpy as np
import timeit

from datetime import datetime
from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import pandas as pd

def run_basics_transformer_model(size, context_length, d_model, d_ff, num_layers, num_heads, w_num_steps, num_steps, use_autocast):
    print(f"=================Benchmark for model {size} {"mixed precision" if use_autocast else "full precision"} started=================", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BasicsTransformerLM(
            vocab_size=10000,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=0.5).to(device)
    
    optimizer = AdamW(model.parameters(), lr=0.01)
    
    x = torch.randint(0, 10000, (4, context_length), device=device)
    y = torch.randint(0, 10000, (4, context_length), device=device)

    for step in range(w_num_steps):
        print(f"\rWarm-up step forward pass: {step}", end="")
        logits = model(x)

    forward_time = []
    backward_time = []
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    for step in range(num_steps):
        print(f"\rBenchmark step forward pass: {step}", end="")
        t1 = timeit.default_timer()
        with autocast_context:
            optimizer.zero_grad()

            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)

            logits = model(x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t2 = timeit.default_timer()
            forward_time.append(t2-t1)

            print(f"\rBenchmark step backward pass: {step}", end="")
            t1 = timeit.default_timer()
            loss = cross_entropy(logits, y)

        loss.backward()
        #optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t2 = timeit.default_timer()
        backward_time.append(t2-t1)

    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot(f"/workspace/memory_snapshot_{size}_{context_length}_{"mixed_precision" if use_autocast else "full_precision"}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.pickle")

    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)

    print(f"\r=================Benchmark for model {size} {"mixed precision" if use_autocast else "full precision"} finished=================", flush=True)
    forward_time_average = np.average(forward_time)
    forward_time_standard_deviation = np.std(np.array(forward_time))
    backward_time_average = np.average(backward_time)
    backward_time_standard_deviation = np.std(np.array(backward_time))

    print(f"Forward pass timing average: {forward_time_average}")
    print(f"Forward pass timing stadard deviation: {forward_time_standard_deviation}")
    print(f"Backward pass timing average: {backward_time_average}")
    print(f"Backward pass timing stadard deviation: {backward_time_standard_deviation}")

    return (size, forward_time_average, forward_time_standard_deviation, backward_time_average, backward_time_standard_deviation, use_autocast)

if __name__ == "__main__":
    results = []

    """ results.append(run_basics_transformer_model(size="small", context_length=256, d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="medium", context_length=256, d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="large", context_length=256, d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="xl", context_length=256, d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="2.7B", context_length=256, d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10, use_autocast=False))

    results.append(run_basics_transformer_model(size="small", context_length=256, d_model=768, d_ff=3072, num_layers=12, num_heads=12, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="medium", context_length=256, d_model=1024, d_ff=4096, num_layers=24, num_heads=16, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="large", context_length=256, d_model=1280, d_ff=5120, num_layers=36, num_heads=20, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="xl", context_length=256, d_model=1600, d_ff=6400, num_layers=48, num_heads=25, w_num_steps = 5, num_steps = 10, use_autocast=True))

    results.append(run_basics_transformer_model(size="2.7B", context_length=256, d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10, use_autocast=True)) """

    results.append(run_basics_transformer_model(size="2.7B", context_length=256, d_model=2560, d_ff=10240, num_layers=32, num_heads=32, w_num_steps = 5, num_steps = 10, use_autocast=True))


    df = pd.DataFrame(results, columns=['Model', 'Forward Time Avg', 'Forward Time Std', 'Backward Time Avg', 'Backward Time Std', 'Mixed precision BF16'])

    print("\n=================Benchmark Results=================")
    print(df.to_markdown(index=False))
    print("===================================================\n")