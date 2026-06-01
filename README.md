# lesson-engine (Chalk)

Lesson generation engine for Bantrly's spoken English coaching platform. Generates structured lesson content across 11 types (tongue twisters, roleplays, poems, etc.) for K-12 age groups, scores output with a rubric-based quality gate, and deduplicates via semantic similarity.

The core experiment: fine-tune a small model (Phi-3 mini 3.8B) on task-specific data and show it matches Gemini 2.5 Flash quality at zero inference cost.

---

## Setup

```bash
uv sync
cp .env.example .env  # add GEMINI_API_KEY and DEEPSEEK_API_KEY
```

## Run

**API server (port 8000):**
```bash
uv run uvicorn api.main:app --reload
```

**Fine-tuned model inference server (port 8001, requires WSL2 + GPU):**
```bash
# from WSL2
python finetune/serve.py
```

**Generate a lesson:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"lesson_type": "tongue_twister", "topic": "S and SH sounds", "age_group": "teen", "difficulty": "hard", "model": "gemini"}'
```

`model` options: `gemini`, `deepseek`, `chalk-phi3`, `ollama/llama3.1:8b`

---

## Reproduce the fine-tune

```bash
# 1. Generate dataset (~$0.22, ~993 examples)
uv run python finetune/prep_dataset.py

# 2. Train (WSL2, RTX 4060 8.6GB, ~1 hour)
python finetune/train.py

# 3. Export to GGUF
python finetune/export_gguf.py
```

---

## Project structure

```
api/                    FastAPI server — /generate, /generate/batch, /lesson-types
config/
  rubric.py             Scoring weights and thresholds for all 11 lesson types
  settings.py           API keys, model names, pricing table
engine/
  clients/              One client per model: gemini, deepseek, ollama, chalk_phi3
  prompts/              Rubric-native prompt builders (one per lesson type)
  generator.py          Orchestrator: prompt → generate → quality gate → retry
  quality_gate.py       Sanity check + LLM-based TA scoring + FAISS dedup
finetune/
  prep_dataset.py       Generates ChatML training data from scraper seeds
  train.py              QLoRA fine-tune (Unsloth, Phi-3 mini, 3 epochs)
  serve.py              Inference server for the fine-tuned adapter (port 8001)
  compare.py            Side-by-side: fine-tuned Phi-3 vs Gemini vs DeepSeek
  export_gguf.py        Convert adapter to GGUF Q8_0
  data/                 train.jsonl, eval.jsonl (993 examples, 85/15 split)
schema/
  lesson_types.py       Pydantic models for all 11 lesson types + enums
scraper/
  raw/                  Curated seed content (11 jsonl files, Gutenberg + hand-picked)
  sources.py            Gutenberg scraper
  pipeline.py           Scrape pipeline
vectorstore/
  dedup.py              FAISS + sentence-transformers semantic dedup
```

---

## Results

| Model | TA Score | Cost/lesson |
|---|---|---|
| Fine-tuned Phi-3 mini | 100% | $0.00 |
| Gemini 2.5 Flash | 100% | ~$0.00013 |
| DeepSeek V3 | 100% | ~$0.00021 (but repeats "She sells seashells") |

Fine-tuned model matches Gemini quality, generates original content, runs at zero marginal cost. Speed on RTX 4060 laptop is ~37s/lesson; production GPU (A100) estimated 2-3s.
