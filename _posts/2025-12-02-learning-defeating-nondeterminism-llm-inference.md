# Notes & Learnings: Defeating Nondeterminism in LLM Inference

Notes & learnings from reading the amazing blog from He, Horace and Thinking Machines Lab, ["Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), Thinking Machines Lab: Connectionism, Sep 2025. Some examples & code & sentences are copied from the original blog, also added with some of my understandings for personal learning purpose.

## Problem the Blog Explained

Why do current LLM inference engines show nondeterministic results even when setting sampling temperature to 0? The blog provides a comprehensive overview of the culprit of nondeterminism in LLM inference and suggests strategies to avoid it.

## "Concurrency + Floating Point" Hypothesis

The blog first reminds me of a situation we noticed before: when we were writing matrix multiplication implementations from scratch (both using CUDA for GPU and using C++ for CPU — we implemented for Intel machines using AMX/AVX intrinsics), we noticed that the results of different implementations sometimes have slight differences. Our hypothesis on this aligned with the "concurrency + floating point" hypothesis mentioned in the blog (different implementations and the nature of GPU parallel computing cause different orders of additions, e.g., `(a + b) + c != a + (b + c)` due to precision and rounding errors).

## Floating Point Addition Order Problems — Concrete Example

How does floating point addition with different orders cause problems?

```python
import numpy as np

a = np.float16(1230)
b = np.float16(23.4)
c = np.float16(23.4)

print("a:", a)
print("b:", b)
print("c:", c)

a_b = a + b
print("a + b:", a_b)
b_c = b + c
print("b + c:", b_c)

d1 = (a + b) + c
d2 = a + (b + c)
print("d1:", d1)
print("d2:", d2)

# Printed results:
# a: 1230.0
# b: 23.4
# c: 23.4
# a + b: 1253.0
# b + c: 46.8
# d1: 1276.0
# d2: 1277.0
```

As mentioned in the blog, when we add two floating-point numbers with different exponents (e.g. 1230 and 23.4), if the results cannot maintain enough digits of precision it will drop last digits.

## The Direct Cause of Nondeterminism in LLM Inference

But how exactly does LLM inference get affected by the "concurrency and floating point hypothesis"? What's the more direct cause of this?

The blog points out that the culprit is: **"The primary reason nearly all LLM inference endpoints are nondeterministic is that the load (and thus batch size) nondeterministically varies! If we'd like to avoid nondeterminism in our inference servers, we must achieve batch invariance in our kernels."**

Now let's dive deep into details on why dynamic batch size causes nondeterminism and how to achieve batch-invariant kernels.

## How Dynamic Batch Size Affects Matrix Multiplication & How to Achieve Batch-Invariant Matrix Multiplication

When we were writing matrix multiplication from scratch, for different MNK dimensions we call different implementations under the hood. Different implementations can use different tiling strategies/configurations, different tensor core instructions, or they can call Split-K implementation for small M & N cases. For the PyTorch matrix multiplication, it follows a similar pattern. M is determined by the online traffic and scheduling. If M is dynamic, when we call the matrix multiplication through PyTorch — which calls cuBLAS under the hood — it will make decisions to pick the matrix multiplication strategy based on M, N, K. If it chooses different strategies, it may break batch invariance.

Some examples of how matrix multiplication can become non-batch-invariant are:

- **Split-K**: For small M and N, if the output tile cannot saturate the GPU, it will decide to do "Split-K". If this happens, it needs to do all-reduce across CUDA blocks, which changes the reduction order compared with other matrix multiplication strategies. Stream-K also has similar issues.
- Choose different tensor core instructions
- Choose different tile sizes

**"So, the easiest way to ensure batch invariance for matrix multiplications is to compile one kernel configuration and use that for all shapes."** — Though this will hurt some performance, it's a trade-off between deterministic LLM inference and better performance. As the example given by the blog:

```python
import torch
torch.set_default_device('cuda') 

B = 2048
D = 4096
a = torch.linspace(-1000, 1000, B*D).reshape(B, D)
b = torch.linspace(-1000, 1000, D*D).reshape(D, D)
# Doing a matrix vector multiplication by taking
# the first element of the batch
out1 = torch.mm(a[:1], b)
# Doing a matrix matrix multiplication and then taking
# the first element of the batch
out2 = torch.mm(a, b)[:1]
print((out1 - out2).abs().max()) # tensor(1669.2500, device='cuda:0')
```

"Doing a matrix multiplication by taking the first element of the batch" gives different results from "doing a matrix multiplication on the full matrix and then taking the first element of the batch". When calling `torch.mm(a[:1], b)`, M = 1, but when calling `torch.mm(a, b)[:1]`, M = B = 2048. I think different M values under the hood call matrix multiplication in different ways (e.g., different tile configurations, or even different matrix multiplication implementations).

## How Dynamic Batch Size Affects Math Operations & How to Achieve Batch-Invariant Reduction Operations

Besides the matrix multiplication kernels, there are math operation kernels in model forward passes (e.g., RMSNorm, LayerNorm, softmax). Those math operations usually involve getting sum/max/etc. The easiest way of getting the sum or max would be using atomic operations, which will cause nondeterministic results because the order is not guaranteed when using atomics. But we usually use reduction instead of atomic operation to get the sum/max; however, for different batch sizes, the implementation can change under the hood:

- If we implement the RMSNorm as each batch only be handled by one CUDA block, the reduction strategy for each batch remains not affected by the batch size.
- When we increase the batch size, it's just each CUDA block sequentially handles more batches, which preserves the batch invariance.
- When we decrease the batch size to smaller than the number of SMs, "If we have a small batch size, our data-parallel strategy may no longer have sufficient parallelism to saturate our cores. In this case, it may be more efficient to 'split' a reduction among multiple cores, allowing us to fully utilize our GPU."

**"The easiest solution is to simply ignore these cases altogether. This is not completely unreasonable — a small batch size means that the kernel is likely to execute quickly anyway, and so a slowdown may not be catastrophic."** But if we want to still parallelize each batch calculation across CUDA blocks, we need to be cautious about the reduction order to be invariant of the batch size. I think implementation like: warp reduction to get each warp's local sum and save to shared memory based on the warp id, say we have 1024 threads, so we need 1024 / 32 = 32 values in shared memory per CUDA block, and let the first 32 threads (first warp) to read the value from shared memory and get the CUDA block sum and save to HBM based on the CUDA block id, say we have 108 CUDA blocks working on each batch, ceil_div(108, 32) = 4, then we only let the first CUDA block's first 4 warps do the final reduction, all of the reductions will be warp level reduction, and by calculating per block sum through shared memory and calculating global sum through HBM, this kind of reduction implementation should be batch-invariant, even each batch is handled by many CUDA blocks.

## How Dynamic Batch Size Affects Attention & How to Achieve Batch-Invariant Attention

As mentioned in the blog, "depending on the inference engine's choices, it's possible that a sequence might get processed in several parts (such as in chunked prefill) or perhaps all at once (if the prefill isn't split up). One example given by the blog looks when using paged KV cache, say block size is 32, we already have 80 tokens (which needs 3 blocks), we then compute 48 tokens (which needs 2 blocks), so 5 blocks in total to compute — which is different reduction order compared with if those KV cache are continuous for the 80 + 48 = 128 tokens which can fit in 4 blocks.

To solve this, the blog says we just need to update the KV cache and page table before attention, ensuring keys & values are consistently laid out regardless of how many tokens are being processed.

Besides the above example, FlashAttention might choose different parallelization strategy based on batch size, e.g., for small batch size, uses Split-KV (parallelize over sequence dimension for GPU utilization), for large batch size, uses regular FlashAttention (sufficient parallelism from batch dimension). The solution given by the blog is: **"In other words, instead of fixing the # of splits, we fix the size of each split and then end up with a varying number of splits. In this manner, we can guarantee that regardless of how many tokens we're processing, we always perform the identical reduction order."**

## Summary

From above discussions, now we can see exactly how the dynamic batch size might make the kernels in model forward pass non-deterministic. If your system requires high determinism, you need batch-invariant kernels. **"In order to achieve batch invariance, it's necessary that the reduction order for a given token does not depend on how many other tokens from its sequence are being simultaneously processed."** Though the tradeoff is performance can get hurt a bit.

## Implementation from the Blog Author's Repo

[https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py](https://github.com/thinking-machines-lab/batch_invariant_ops/blob/main/batch_invariant_ops/batch_invariant_ops.py)

The implementation uses `torch.Library` to substitute PyTorch operators in an unintrusive way:

```python
_batch_invariant_LIB = torch.library.Library("aten", "IMPL")
_batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA") 
_batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")
```

This allows the library to intercept standard PyTorch operations and replace them with deterministic versions without modifying the model code.

## Batch-Invariant Matmul

The core of the matrix multiplication implementation uses a persistent kernel strategy with fixed configuration. Uses a persistent kernel approach where each SM processes multiple tiles in a deterministic order; if batch size is small, the number of CUDA blocks (grid size) launched is still fixed — some CUDA block can be idle if number of tiles is smaller than number of SM. The persistent kernel strategy avoids the dynamic strategy like choosing Split-K when batch size is small, which might break batch-invariant. I think persistent kernel implementation is a clean way to give deterministic behavior that ensures consistent execution order.

```python
# Small matrix (100x100): 1 tile total
# - Block 0 processes tile 0
# - Blocks 1-107 do nothing

# Large matrix (1000x1000): 64 tiles total  
# - Block 0 processes tiles: 0, 108, 216, ... (every 108th tile)
# - Block 1 processes tiles: 1, 109, 217, ... 
# - Same deterministic pattern regardless of matrix size
```

In the persistent kernel implementation, the `_compute_pid` function ensures consistent mapping of tiles to processing units regardless of batch size:

```python
@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n
```

Uses predetermined tile sizes based on dtype rather than adaptive selection. The fixed tile size is very important for ensuring batch-invariant, because, as discussed above, using different tile size gives different reduction order; also, fixed tile size also should ensure under the hood using the same tensor core instruction.

```python
configs = {
    torch.bfloat16: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
    },
    torch.float16: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
    },
    torch.float32: {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8, "num_stages": 3, "num_warps": 8,
    },
}
```

## Batch-Invariant Reduction Operations

Calculate mean using Triton they provided as below. Each row's mean is handled by one CUDA block, and calculating the sum in fixed order ensures the batch-invariance.

```python
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,  # size before reduction dim
    N,  # size of reduction dim
    K,  # size after reduction dim
    BLOCK_SIZE: tl.constexpr,
):
    """
    Kernel for computing mean along a single dimension.
    Input is viewed as (M, N, K) where N is the dimension being reduced.
    """
    # Program ID gives us which output element we're computing
    pid = tl.program_id(0)

    # Compute output indices
    m_idx = pid // K
    k_idx = pid % K

    # Bounds check
    if m_idx >= M or k_idx >= K:
        return

    # Accumulate sum across reduction dimension
    acc = 0.0
    for n_start in range(0, N, BLOCK_SIZE):
        n_offsets = n_start + tl.arange(0, BLOCK_SIZE)
        mask = n_offsets < N

        # Calculate input indices
        input_idx = m_idx * input_stride0 + n_offsets * input_stride1 + k_idx * input_stride2

        # Load and accumulate
        vals = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
        acc += tl.sum(vals)

    # Compute mean and store
    mean_val = acc / N
    output_idx = m_idx * output_stride0 + k_idx * output_stride1
    tl.store(output_ptr + output_idx, mean_val)
```

Compute Log Softmax batch-invariant code from the author's repo, each block handles one row of the input tensor, and consistent reduction patterns ensure batch-invariance:

```python
@triton.jit
def _log_softmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute log_softmax along the last dimension of a 2D tensor.
    Each block handles one row of the input tensor.
    """
    # Get the row index for this block
    row_idx = tl.program_id(0).to(tl.int64)

    # Compute base pointers for input and output rows
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride

    # Step 1: Find maximum value in the row for numerical stability
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))

        # Update maximum
        max_val = tl.max(tl.maximum(vals, max_val))

    # Step 2: Compute sum of exp(x - max_val)
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)

        # Compute exp(x - max_val) and accumulate
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))

    # Compute log(sum_exp)
    log_sum_exp = tl.log(sum_exp)

    # Step 3: Compute final log_softmax values: x - max_val - log_sum_exp
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols

        # Load values
        vals = tl.load(row_start_ptr + col_idx, mask=mask)

        # Compute log_softmax
        output = vals - max_val - log_sum_exp

        # Store results
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)
```

## Reference

- He, Horace and Thinking Machines Lab, ["Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), Thinking Machines Lab: Connectionism, Sep 2025.
- My original blog: [https://medium.com/@yuezhang2455/llm-blog-learning-thinking-machine-labs-defeating-nondeterminism-in-llm-inference-8c059846d30f](https://medium.com/@yuezhang2455/llm-blog-learning-thinking-machine-labs-defeating-nondeterminism-in-llm-inference-8c059846d30f), copied to here as I will maintain blogs here going forward.

