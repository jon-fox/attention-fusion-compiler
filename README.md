## The What

A compiler that takes different attention variants (causal masking, sliding window, ALiBi) and automatically generates a single fused GPU kernel for each one, instead of requiring a hand written kernel per variant.

## The problem

New attention mechanisms get published often, but turning a new variant into a fast, fused GPU kernel takes real engineering time. Most new ideas run slow and unfused for a while before someone optimizes them by hand. This project builds a small, honest version of a compiler that automates that step.

## The How

1. Define a custom MLIR dialect that represents attention operations (matmul, softmax, masking, scaling) as a graph.
2. Write fusion passes that combine these operations into one kernel, avoiding unnecessary trips to GPU memory.
3. Lower the fused representation to a GPU backend using Triton, which compiles it down to executable GPU code.
4. Benchmark the generated kernel against a naive unfused version and against PyTorch's built in attention implementation.

## The Goals

- Support at least three attention variants through the same dialect and passes, not one fixed kernel.
- Prove the fusion passes actually help by showing real benchmark numbers, not just working code.
- Keep the fusion logic in the compiler passes, using Triton only as the final code generation step.

## The not

This is not a claim to beat or replace existing tools like PyTorch's FlexAttention. It is a smaller version of the same kind of problem, built to demonstrate real compiler engineering: a custom IR, real optimization passes, and a working lowering path to GPU code.

## The Status

Early stage. Correctness and passes are being developed and tested locally. GPU lowering and benchmarking require a CUDA capable GPU and are run on rented cloud hardware.

## The Hardware
GPU specific needs can be executed on runpod or similar hosting service

