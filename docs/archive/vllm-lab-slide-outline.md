# vLLM vs Baseline Comparison - Slide Deck Outline (2-3 ชม. Session)

ประมาณ 18-22 slide เน้น lab มากกว่า lecture เพราะ retention มาจากมือทำ ไม่ใช่จากสไลด์
Pacing reference: lecture slide รวมไม่เกิน 45-50 นาที ส่วนที่เหลือเทไปที่ lab + discussion

## 1. Hook (2-3 slides)

- Hook question: "ถ้าโมเดลเดียวกัน เครื่องเดียวกัน ทำไมการเลือก serving engine ถึงเปลี่ยนได้ว่าระบบจะรองรับ user ได้ 10 คน หรือ 200 คนพร้อมกัน?"
- Agenda
- Learning objective

## 2. Problem Framing (2 slides)

- Naive serving bottleneck คืออะไร
- GPU idle time diagram ของ static batching

## 3. Core Concept (5-6 slides)

- PagedAttention (OS virtual memory paging analogy)
- Continuous batching diagram (timeline ของ GPU utilization)
- KV cache fragmentation: before/after
- สรุป 5 key message เป็น 1 slide ตาราง (ดูตารางด้านล่าง)

### 5 Key Message Summary Table (ใช้เป็น slide สรุป)

| Message | เหตุผลเชิงเทคนิค | Demo จาก Lab ที่พิสูจน์ |
|---|---|---|
| 1. Concurrency = จุดที่ vLLM ชนะขาด | Continuous batching แทรก request ใหม่เข้า batch ได้ทันที ไม่ต้องรอ batch เก่าจบ | Experiment B: throughput vs concurrency graph |
| 2. Memory management ฉลาดกว่า | PagedAttention ตัด KV cache เป็น block non-contiguous ลด fragmentation | VRAM usage ก่อน OOM เทียบสอง engine |
| 3. Cost-per-token ต่ำกว่าใน production | Throughput สูงขึ้นบน hardware เดิม = ต้นทุนต่อ request ลดลง | คำนวณ tokens/sec/$ จากผล Experiment B |
| 4. Operational ง่ายขึ้น | OpenAI-compatible API ทำให้ client code ไม่ต้องเปลี่ยน | โชว์ request ผ่าน `/v1/chat/completions` |
| 5. Accuracy ไม่เปลี่ยน (control message) | weight เดิม, logits เดิม (modulo numerical precision) | Experiment C: diff output ระหว่างสอง engine |

## 4. Lab Brief (4-5 slides)

- Environment setup (Colab/Kaggle, model เลือก, GPU requirement)
- Experiment A instruction (single request latency)
- Experiment B instruction (concurrent throughput)
- Experiment C instruction (accuracy diff)
- Deliverable ที่ต้องเก็บ (CSV/graph)

### Lab Design Rationale

**Approach:** สร้างโปรเจคเดียวกัน 2 version (with/without vLLM) แล้วเทียบกัน 3 dimensions:

| Exp | วัด | วิธี |
|-----|-----|------|
| A | Single request latency | ส่ง request ทีละอัน เทียบ time-to-first-token |
| B | Concurrent throughput | ส่งพร้อมกัน N users เทียบ tokens/sec |
| C | Accuracy diff | output เหมือนกันไหม บน weight เดิม |

จุดขายหลักคือ **Experiment B** - concurrency คือที่ vLLM ชนะชัดที่สุด กราฟที่คาดไว้:

```
tokens/sec
    |              vLLM ──────────────────────────────
    |                          /
    |                        /
    |                      /
    |    HF Baseline ──────── (plateau หรือ OOM ก่อน)
    |
    +──────────────────────────── concurrent users
         1   5   10   20   50
```

บน T4, baseline จะ plateau หรือ crash ก่อน vLLM ยังไม่ถึงขีดจำกัด = visual proof ที่เด็กจำได้

**Free-tier framing (pedagogical key message):**
- vLLM ได้เปรียบจาก algorithm ไม่ใช่ hardware
- ถ้า gap โผล่แม้แต่บน free tier, คนฟัง "believe" ได้ทันที และ verify เองได้หลังคลาส
- ถ้า hardware ดีขึ้น, throughput ขยายทั้งคู่ แต่ slope ของ vLLM สูงกว่า (multiplicative ไม่ใช่ additive)
- ใช้ฟรีทีเออร์เพื่อพิสูจน์ว่า "แม้บน constrained hardware ก็เห็น gap แล้ว production ยิ่ง exponential"

### Environment Setup Detail

**Target platform:** Google Colab free tier (T4 GPU)

| Component | Choice | เหตุผล |
|-----------|--------|--------|
| GPU | T4 (15GB VRAM) | free tier NVIDIA GPU เดียวที่รัน vLLM ได้ |
| Model | `Qwen2.5-1.5B-Instruct` | fit T4 สบาย (~3-4GB), quality ดีพอสาธิต |
| Baseline | HF Transformers + asyncio sequential | ง่าย, เห็น bottleneck ชัด |
| vLLM server | OpenAI-compatible API | student ใช้ `openai` client เดิมได้ |
| Load test | `asyncio.gather()` N concurrent requests | ไม่ต้อง install tools พิเศษ |
| Output | CSV + matplotlib | รัน in-notebook ได้เลย |

**GPU selection note (สำหรับ lab instruction):**
- ต้องเปลี่ยน runtime เป็น T4 GPU ก่อนเสมอ (default คือ CPU)
- v5e TPU: vLLM ไม่ support TPU natively - ห้ามเลือก
- CPU only: รัน vLLM ไม่ได้, baseline ช้ามากจน benchmark ไม่ meaningful
- ระบุในหน้าแรก notebook ให้ชัด เพราะเด็กจะลืมแล้วได้ `CUDA not available` error

## 5. Hands-on (ไม่มี slide ระหว่างนี้)

- 75-90 นาที ลงมือจริง ไม่ใช้สไลด์

## 6. Result Sharing & Discussion (3-4 slides)

- Template graph ว่างให้แต่ละกลุ่ม fill ผลตัวเอง
- Decision matrix: "เมื่อไหร่ใช้ vLLM เมื่อไหร่ไม่จำเป็น"
- เทียบ vLLM vs TGI / SGLang / TensorRT-LLM แบบย่อ (model compatibility, license, peak throughput, vendor lock-in)

## 7. Wrap-up (1-2 slides)

- Recap 5 key message
- Reading / resource เพิ่มเติม

## Timing Summary

| Section | เวลา |
|---|---|
| Hook + Problem Framing | 10-15 นาที |
| Core Concept | 20-25 นาที |
| Lab Brief | 15-20 นาที |
| Hands-on | 75-90 นาที |
| Result Sharing & Discussion | 20-25 นาที |
| Wrap-up | 5-10 นาที |
