# Alternative Models for vLLM Concurrency Lab

Drop-in replacements for `Qwen/Qwen2.5-1.5B-Instruct`

## How to Use

1. Copy a Model ID from the table below
2. In `lab.ipynb`, find this line in the Setup cell:
   ```python
   MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
   ```
3. Replace the value with your chosen Model ID, then rerun all cells

> **GATED models** require accepting the license on the HuggingFace model page and running `huggingface-cli login` before the notebook. Not recommended for quick workshop use.

---

## Model List

### Currently Used

| Model ID | Params | Notes |
|----------|--------|-------|
| `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | Default. Instruction-tuned, float16 friendly |

---

### Lighter - use if VRAM is tight

| Model ID | Params | Notes |
|----------|--------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | Smallest in Qwen2.5 family. Same chat template, cleanest swap. Outputs shorter/weaker but benchmarks still valid |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B | Reliable for concurrency benchmarking. Different chat template - outputs may look different |

---

### Similar Size - good alternatives

| Model ID | Params | Notes |
|----------|--------|-------|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | **RECOMMENDED** alternative. Very efficient for its size. No login required |
| `Qwen/Qwen2-1.5B-Instruct` | 1.5B | Previous generation Qwen. Nearly identical resource usage. Good for comparing same-size different-generation behavior |

---

### Slightly Heavier - richer outputs

| Model ID | Params | Notes |
|----------|--------|-------|
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | Strong reasoning for size. Uses more VRAM (~7-8GB). May need `--gpu-memory-utilization 0.70` in vLLM cell |

---

### GATED - requires HF login + license accept

| Model ID | Params | Notes |
|----------|--------|-------|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Lightest Llama 3.2. Good quality. Requires HF token |
| `google/gemma-2-2b-it` | 2B | Google Gemma 2, instruction-tuned. Requires HF token + Google license acceptance |

---

## VRAM Reference (T4 = 15GB)

| Model Size | Weight Usage | Status |
|-----------|-------------|--------|
| 0.5B | ~1-2GB | Very safe, plenty of KV cache room |
| 1.1B | ~2-3GB | Safe |
| 1.5B | ~3-4GB | Safe (default lab setting) |
| 1.7B | ~3-4GB | Safe |
| 2B | ~4-5GB | Safe |
| 3.8B | ~7-8GB | Reduce `--gpu-memory-utilization` to `0.70` |

---

## VRAM Calculator

### Quick Formula

```
Weight VRAM (GB) = (Params_B × Bits) / 8
Total needed    = Weight VRAM × 1.2   ← add ~20% for KV cache + runtime overhead
```

Example: 7B model at FP16 → `(7 × 16) / 8 = 14 GB` weights + ~3 GB overhead = ~17 GB total

---

### Weight Size by Precision

| Model Size | FP16 / BF16 (16-bit) | INT8 (8-bit) | INT4 (4-bit) |
|-----------|----------------------|--------------|--------------|
| 0.5B | ~1 GB | ~0.5 GB | ~0.3 GB |
| 1.5B | ~3 GB | ~1.5 GB | ~0.8 GB |
| 3B | ~6 GB | ~3 GB | ~1.5 GB |
| 7B | ~14 GB | ~7 GB | ~3.5 GB |
| 13B | ~26 GB | ~13 GB | ~6.5 GB |
| 32B | ~64 GB | ~32 GB | ~16 GB |
| 70B | ~140 GB | ~70 GB | ~35 GB |

> Values are weight-only. Add 20-30% headroom for KV cache during inference.

---

### Sweet Spot by GPU

| GPU VRAM | Comfortable (FP16) | Push it (INT4) | vLLM flag |
|----------|--------------------|----------------|-----------|
| 6 GB (RTX 3060 / 4060) | 1.5-3B | 7B | `--gpu-memory-utilization 0.70` |
| 8 GB (RTX 3070 / 4060 Ti) | 3B | 7B (tight) | `--gpu-memory-utilization 0.75` |
| 12 GB (RTX 3080 / 4070) | 7B | 13B | default `0.90` ok |
| 15 GB (T4 - this lab) | 7B | 13B | default `0.90` ok |
| 16 GB (RTX 4080) | 7B | 13B | default `0.90` ok |
| 24 GB (RTX 3090 / 4090) | 13B | 32B | default `0.90` ok |

**Rule:** if the model fits comfortably at FP16, use FP16 - quantization trades quality for VRAM, not speed alone.

---

### Reverse Lookup - Given my VRAM, what can I run?

**8 GB** (RTX 4060, RTX 3070)
- Safe: 1.5-3B at FP16
- Push: 7B-8B at INT4 only (~4.5 GB weights + ~3.5 GB headroom)
- Recommended: `Qwen/Qwen2.5-1.5B-Instruct` (FP16) or `Llama-3-8B-Instruct` (Q4)

**16 GB** (T4 Colab, RTX 4060 Ti 16GB)
- Safe: 7B-8B at FP16 (tight - reduce `--max-model-len` to 2048)
- Push: 14B at INT4/INT8
- Recommended: `Qwen/Qwen2.5-7B-Instruct` (FP16) or `Qwen2.5-14B-Instruct` (Q4)

**24 GB** (RTX 3090, RTX 4090, A5000)
- Safe: 7B-8B at FP16 with full context + high concurrency
- Push: 32B at INT4 (~18 GB weights, ~6 GB headroom - works for short context)
- Recommended: `Llama-3-8B-Instruct` (FP16, production throughput) or `Qwen2.5-32B-Instruct` (Q4)

**48 GB+** (A6000, 2x RTX 3090/4090 NVLink)
- Safe: 32B at FP16 or 70B at INT4 (~40 GB weights + ~8 GB headroom)
- Recommended: `Llama-3-70B-Instruct` (Q4) or `Qwen2.5-72B-Instruct` (INT4)
