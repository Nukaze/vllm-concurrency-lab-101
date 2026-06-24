# vLLM Concurrency Lab 101

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nukaze/vllm-concurrency-lab-101/blob/main/lab.ipynb)

A 2-3 hour hands-on lab comparing HF Transformers naive serving vs vLLM on the same model and hardware.

**Key question:** If the model weights are identical, why does choosing the serving engine change whether your system handles 10 users or 200 users simultaneously?

---

## What you will measure

| Exp | Metric | What to look for |
|-----|--------|-----------------|
| A | Single request latency | Baseline vs vLLM time-to-completion |
| B | Concurrent throughput | Throughput curve as N users increases - where does baseline plateau? |
| C | Output accuracy diff | Are outputs near-identical on same weights? |

---

## Setup

1. Click **Open In Colab** above
2. Go to **Runtime > Change runtime type > T4 GPU** (required - vLLM needs CUDA)
3. Run all cells top to bottom

> CPU or TPU runtimes will not work. v5e TPU is not supported by vLLM.

---

## Stack

| Component | Choice |
|-----------|--------|
| GPU | Google Colab T4 (free tier, 15GB VRAM) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Baseline | HF Transformers - sequential, no batching |
| vLLM | OpenAI-compatible server (`vllm serve`) |
| Load test | `asyncio.gather()` - concurrent requests |

---

## Repo structure

```
vllm-concurrency-lab-101/
├── lab.ipynb                # Main lab notebook (run this in Colab)
├── vllm-lab-slide-outline.md  # Slide deck outline for instructors
└── .gitignore
```

---

## Expected results (T4 free tier)

- **Exp A:** vLLM typically 1.2-2x faster on single requests due to optimized CUDA kernels
- **Exp B:** Baseline throughput flattens or degrades beyond 4-8 concurrent users; vLLM continues scaling
- **Exp C:** Similarity score > 0.95 - same weights, near-identical outputs

The gap widens on larger GPUs (A100, H100) - free tier proves the algorithm advantage before hardware scaling amplifies it.
