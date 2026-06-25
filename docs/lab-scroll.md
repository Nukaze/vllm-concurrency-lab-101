# Lab Scroll - เฉลยคำตอบ (Instructor Reference)

---

## Cell 7 - baseline_generate

```python
t0 = time.perf_counter()  # จับเวลาเริ่มต้น ก่อนที่ GPU จะเริ่มคำนวณ

output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)  # รัน autoregressive decoding บน GPU, do_sample=False = greedy (ไม่สุ่ม)

t1 = time.perf_counter()  # จับเวลาสิ้นสุด หลัง GPU generate ครบทุก token แล้ว
```

`perf_counter` ให้ precision สูงกว่า `time.time()` เหมาะกับการวัด latency ระดับ ms
`do_sample=False` = greedy decoding ผลลัพธ์ deterministic ทุกครั้ง สำคัญสำหรับ Exp C ที่ต้องเทียบ output

---

## Cell 9 - Exp B baseline loop

```python
_, _, n_tok = baseline_generate(prompt)  # เรียก baseline สร้าง output, รับเฉพาะ n_tokens กลับมา (ทิ้ง text และ latency)
total_tokens += n_tok  # สะสม token รวมของทุก request ใน batch นี้ เพื่อคำนวณ throughput ทีหลัง
```

loop นี้ประมวลผลทีละ request (sequential) ดังนั้น throughput แทบไม่เพิ่มเมื่อ N มากขึ้น
นี่คือจุดที่ทำให้กราฟ baseline flat ใน Exp B

---

## Cell 15 - vllm_generate

```python
resp = client.chat.completions.create(  # ส่ง HTTP POST ไปยัง vLLM server ที่ localhost:8000 (OpenAI-compatible API)
    model=MODEL_ID,  # ชื่อ model ที่ serve อยู่บน server ต้องตรงกับที่ vLLM โหลดไว้
    messages=[{'role': 'user', 'content': prompt}],  # prompt ในรูปแบบ chat message เหมือน ChatGPT API
    max_tokens=MAX_NEW_TOKENS,  # จำกัด output tokens ให้เท่ากับ baseline เพื่อเทียบกันได้
    temperature=0,  # greedy decoding ไม่สุ่ม ผลลัพธ์ deterministic เหมือน do_sample=False
)
```

`temperature=0` เทียบเท่า `do_sample=False` ใน HF - API ต่างกันแต่ความหมายเหมือนกัน
`api_key='dummy'` เพราะ vLLM ที่ localhost ไม่มี auth แต่ OpenAI client บังคับให้ใส่

---

## Cell 17 - run_concurrent

```python
token_counts = await asyncio.gather(*[vllm_generate_async(p) for p in batch])  # สร้าง coroutine ทุกตัวแล้วส่งพร้อมกัน, await รอจนทุก request ตอบกลับครบ
```

`*[...]` คือ unpack list เป็น positional args เพราะ `gather()` รับ coroutines แยกตัว ไม่ใช่ list
ทุก request ถูกส่งพร้อมกัน vLLM รับแล้วจัดการผ่าน continuous batching ทำให้ throughput เพิ่มตาม N

---

## Key Takeaways

| TODO | Concept | สิ่งที่ควรจำ |
|------|---------|-------------|
| Cell 7 - timing + generate | Benchmarking precision | `perf_counter` วัด latency ได้แม่นกว่า `time.time()`, `do_sample=False` ทำให้ผล reproducible |
| Cell 9 - sequential loop | Sequential vs throughput | serving แบบ loop = throughput flat ไม่ว่า N จะเพิ่มแค่ไหน คือ bottleneck ของ naive server |
| Cell 15 - OpenAI API call | API abstraction | HF และ vLLM ใช้ parameter คนละชื่อแต่ความหมายเหมือนกัน - engine เปลี่ยน client code แทบไม่ต้องเปลี่ยน |
| Cell 17 - asyncio.gather | Concurrency model | `gather` = ส่งพร้อมกัน, `await` ใน loop = ทีละตัว - ต่างกันคือ throughput ทั้งหมด |

**Big picture:** weights เหมือนกัน แต่ engine เปลือง resource ต่างกัน
HF ทำ 1 request แล้ว GPU ว่าง รอ request ถัดไป
vLLM ใช้ KV cache block + continuous batching ทำให้ GPU ไม่ว่างเลย throughput จึง scale
