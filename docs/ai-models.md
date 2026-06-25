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
