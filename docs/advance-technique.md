# Advanced vLLM Techniques -- For the Curious

What you measured in the lab (PagedAttention + continuous batching) is the **floor**, not the ceiling.
These are the next levers to pull.

---

## 1. Speculative Decoding

A small "draft" model generates several tokens ahead. The large "target" model verifies them all in one forward pass.
Correct guesses are accepted for free -- wrong ones get corrected at the first mismatch.

- Gain: 2-3x latency reduction on top of existing vLLM gains
- Best for: long outputs where the draft model guesses well (code, structured text)
- vLLM flag: `--speculative-model <draft_model_id>`

Search: `vLLM speculative decoding`, `Medusa decoding`, `EAGLE speculative decoding`

---

## 2. Quantization (AWQ / GPTQ / FP8)

Reduce weight precision from float16 to int4/int8/fp8.
Smaller weights = less VRAM = more KV cache headroom = more concurrent requests before OOM.

- Gain: 1.5-2x throughput, model fits on smaller GPU
- Trade-off: tiny accuracy loss (usually negligible at int8, noticeable at int4 for small models)
- vLLM flag: `--quantization awq` or `--quantization gptq`

Search: `AWQ quantization LLM`, `GPTQ vs AWQ`, `FP8 inference`

---

## 3. Prefix Caching

If many requests share the same system prompt or few-shot examples, vLLM can cache those KV blocks
and reuse them across requests -- the shared prefix costs zero compute after the first request.

- Gain: near-zero TTFT (time to first token) for requests with matching prefix
- Best for: chatbots, RAG pipelines, any workload with a fixed system prompt
- vLLM flag: `--enable-prefix-caching`

Search: `vLLM prefix caching`, `KV cache reuse LLM serving`

---

## 4. Tensor Parallelism (Multi-GPU)

Split the model's weight matrices across N GPUs. Each GPU handles 1/N of the computation per layer,
then all-reduce to sync. Throughput scales near-linearly with GPU count.

- Gain: ~Nx throughput where N = number of GPUs
- Requirement: GPUs connected via NVLink or high-bandwidth interconnect
- vLLM flag: `--tensor-parallel-size N`

Search: `tensor parallelism transformer`, `Megatron-LM model parallel`

---

## 5. Chunked Prefill

Under heavy concurrent load, long prompts (prefill phase) can block short requests from being decoded.
Chunked prefill breaks long prefills into smaller chunks interleaved with decode steps.

- Gain: reduces tail latency and P99 response time under mixed workloads
- Best for: production serving with mixed prompt lengths
- vLLM flag: `--enable-chunked-prefill`

Search: `chunked prefill vLLM`, `prefill decode disaggregation`

---

## 6. Pipeline Parallelism (Multi-Node)

For models too large for a single node, split layers across nodes.
Node 0 runs layers 0-N/2, node 1 runs N/2-N, with pipeline stages overlapping.

- Gain: enables 70B+ models on commodity hardware
- Trade-off: adds pipeline bubble latency
- vLLM flag: `--pipeline-parallel-size N`

Search: `pipeline parallelism LLM`, `GPipe`, `PipelineParallel vLLM`

---

## Quick Reference -- vLLM Flags

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --speculative-model Qwen/Qwen2.5-0.5B-Instruct \
  --quantization awq \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90
```

## Where to Go Next

- vLLM docs: https://docs.vllm.ai
- vLLM GitHub: https://github.com/vllm-project/vllm
- LLM Inference survey: search `"LLM inference optimization survey 2024"`
- PagedAttention paper: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
