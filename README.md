# vLLM Concurrency Lab 101

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Nukaze/vllm-concurrency-lab-101/blob/main/lab.ipynb)

Lab ลงมือทำ 2-3 ชั่วโมง เปรียบเทียบสองวิธีในการ serve โมเดล AI บนเครื่องและโมเดลเดียวกัน -- HF Transformers (วิธีทั่วไป) กับ vLLM ที่ออกแบบมาสำหรับรองรับ concurrent users จำนวนมาก

**คำถามหลัก:** ถ้าใช้โมเดล AI ตัวเดียวกัน เครื่องเดียวกัน ทำไมการเลือก serving engine ถึงเปลี่ยนได้ว่าระบบจะรองรับ user ได้ 10 คน หรือ 200 คนพร้อมกัน?

---

## สิ่งที่จะทดลองใน Lab

| Experiment | Metric (วัดอะไร) | สิ่งที่ต้องสังเกต |
|------------|-----------------|-----------------|
| A | Single Request Latency (ความเร็วต่อ 1 request) | Baseline vs vLLM ใครตอบกลับเร็วกว่า |
| B | Concurrent Throughput (1, 2, 4, 8, 16 concurrent users) | Baseline เริ่ม plateau หรือ OOM ที่ N เท่าไหร่ |
| C | Output Accuracy Diff (ความแม่นยำของ output) | Same weights -- outputs เหมือนกันไหม |

---

## วิธีเริ่มต้น

1. กด **Open In Colab** ด้านบน
2. บันทึก notebook ลง Google Drive ก่อน: **File > Save a copy in Drive**
   (ถ้าไม่บันทึกแล้ว runtime หลุด จะต้องเริ่มใหม่ทั้งหมด)
3. เปลี่ยนประเภท runtime เป็น GPU: **Runtime > Change runtime type > T4 GPU**
   (จำเป็น -- vLLM ต้องใช้ CUDA GPU เท่านั้น ใช้ CPU หรือ TPU ไม่ได้)
4. Run cell ทั้งหมดจากบนลงล่าง

> **หมายเหตุ:** v5e TPU และ CPU runtime ใช้กับ lab นี้ไม่ได้

---

## Stack ที่ Lab ใช้

| Component | รายละเอียด |
|-----------|------------|
| GPU | Google Colab T4 (free tier, VRAM 15GB) |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` (ใช้ตัวเดียวกันทั้งสอง engine) |
| Baseline | HF Transformers -- sequential, ประมวลผลทีละ request ไม่มี batching |
| vLLM | OpenAI-compatible server (`vllm serve`) |
| Load test | `asyncio.gather()` -- ส่ง concurrent requests พร้อมกัน |
| Visualization | กราฟ matplotlib ภายใน notebook |

---

## โครงสร้างไฟล์

```
vllm-concurrency-lab-101/
├── lab.ipynb               -- notebook หลัก (run บน Colab)
└── docs/
    ├── vllm-lab-slide-outline.md  -- outline สไลด์สำหรับผู้สอน
    ├── interact.txt               -- draft ฟีเจอร์ interactive ที่ยังอยู่ระหว่างออกแบบ
    └── handover.txt               -- บันทึก session ก่อนหน้า
```

---

## ผลที่คาดว่าจะเห็น (บน T4 free tier)

- **Exp A:** vLLM เร็วกว่า baseline ประมาณ 1.2-2x ต่อ 1 request
- **Exp B:** Baseline throughput เริ่ม plateau หรือ OOM เมื่อ concurrent users ถึง 4-8 คน ขณะที่ vLLM ยัง scale ต่อได้
- **Exp C:** Similarity score > 0.95 -- same weights, outputs แทบไม่ต่างกัน

ช่องว่างระหว่างสอง engine จะยิ่งเห็นชัดขึ้นบน GPU ที่ดีกว่า (A100, H100) -- free tier พิสูจน์ได้แล้วว่า vLLM ได้เปรียบจาก algorithm ไม่ใช่จาก hardware

---

## Run Lab บน Windows (ไม่ใช้ Colab)

สำหรับคนที่อยากรันบนเครื่องตัวเองแทน -- ดูขั้นตอนได้ใน **Appendix** ท้าย notebook
ต้องใช้ WSL2 + NVIDIA GPU ที่มี VRAM อย่างน้อย 6GB (RTX 3060 / RTX 4060 ขึ้นไป)
