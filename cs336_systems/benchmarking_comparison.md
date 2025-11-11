# Benchmarking results

## With warmup steps

![benchmarking with warmup steps](benchmarking_script_b_result.png)

## Without warmup steps

![benchmarking without warmup steps](benchmarking_script_c_result.png)

## Conclusions

- Forward pass takes between 0.029 - 0.1 seconds. 
- From the results above we can notice that forward pass takes in average 1.2-2 times more than forward pass.
- Standard deviation for the forward pass is very small ~ 1%, and for backward pass ~ 10%
- In case absence of warmup steps average time increase for both forward and backwards passes around 1.2-1.5 times. The reason for that is probably the model hasn't been loaded into the memory.