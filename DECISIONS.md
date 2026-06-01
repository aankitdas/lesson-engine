# Chalk — Decisions Log

## Project Brief
Standalone demo. Research-backed lesson generation engine for Bantrly.
Proves: fine-tuned small model (Phi-3 mini 3.8B) ≥ large expensive model
for narrow spoken English lesson generation task.
Core thesis: task-specific fine-tuned specialist > general large model for K-12
spoken English lesson generation. Judged by Bantrly's own 6-metric rubric scorer.

## Stack
- Runtime: Python (uv), Python 3.11
- Backend: FastAPI (api/main.py), port 8000
- Generation (primary): Gemini 2.5 Flash API (google-genai SDK)
- Generation (comparison): DeepSeek V3 API (OpenAI-compatible client)
- Local inference: Ollama (llama3.1:8b, mistral:7b, llama3.2:3b)
- Local fine-tuning: Phi-3 mini 3.8B via Unsloth + QLoRA
- Fine-tuned serving: finetune/serve.py (FastAPI, port 8001, OpenAI-compatible)
- Fine-tuned client: engine/clients/chalk_phi3.py → calls localhost:8001/v1/chat/completions
- DGX fine-tuning: Mistral 7B, Llama 3.1 8B (later)
- Embeddings/Dedup: sentence-transformers (all-MiniLM-L6-v2) + FAISS (local)
- DB: SQLite (demo only, not yet wired)
- UI: Single-page HTML/CSS/JS demo showcase

## Domain
- Spoken English coaching: pronunciation, fluency, prosody, vocal delivery
- Age groups: child (≤12), teen (13-15), youth (16-18), adult (19+)
- 11 lesson types across interactive and non-interactive categories
- 6 voice metrics: P, Pr, R, Ff, V, TA (weights are module-specific)

## Lesson Types
Interactive (topic + ai_lines):
  - silly_topic_debate, roleplay, rapid_fire_qa

Non-interactive (student reads/speaks raw content):
  - paragraph_reading, tongue_twister, poem_recitation,
    story_building, yes_and_improv, no_but_improv,
    short_speech, audio_postcard

## Schema
- Typed Pydantic model per lesson type (11 types) in schema/lesson_types.py
- Intentionally diverges from Bantrly's content_key/content_value pattern
- ChalkMeta block on every lesson: model, cost, scores, dedup hash, quality gate result
- GenerationRequest is the teacher-facing input model
- ContentType is a Union of all 11 typed content models
- Token counts stored in rubric_scores dict (input_tokens, output_tokens keys)

## Rubric Strategy
- Rubric-native generation: dominant metric(s) per module injected as generation constraints
- Dominant metrics computed from weight table, ties included (e.g. paragraph_reading → P + Pr)
- Secondary metrics above 0.15 weight also included in prompt guidance
- config/rubric.py is single source of truth for all weights, bands, age modes

## Prompt Architecture
- engine/prompts/base.py: metric guidance builder, age tone, difficulty guidance, teacher context
- engine/prompts/templates.py: 11 prompt builder functions + TEMPLATE_MAP router
- Each builder returns (system: str, user: str) tuple
- Output: JSON matching the typed content schema for that lesson type
- Token-efficient: only dominant + secondary (>0.15) metrics included per prompt

## Quality Gate (M5)
Simplified — only what's meaningful without audio data:
- Sanity check: type-aware minimum word counts (tongue_twister min=4, default min=10), max=800
- TA via LLM (Gemini, temp=0.1): did output follow format, age group, difficulty, topic
- Semantic dedup: FAISS + sentence-transformers, cosine similarity > 0.85 = reject
- Pass threshold: TA >= 0.75 AND not duplicate
- Max retries: 3 before returning best attempt with passed=False
- Heuristic P/Pr/R/Ff/V scoring dropped — requires real speech audio to be meaningful

## Data Pipeline (M2)
Sources:
  - Gutenberg (gutendex.com): paragraph_reading, poem_recitation, story_building
    - Correct param: search (not topic)
    - mime_type=text%2Fplain filter applied
    - \r\n line endings normalised before splitting
  - Curated seeds: all other 8 types (hand-picked, K-12 appropriate)

Raw scrape counts (scraper/raw/*.jsonl):
  | Type               | Count | Source           |
  |--------------------|-------|------------------|
  | paragraph_reading  |    97 | Gutenberg + seed |
  | poem_recitation    |    53 | Gutenberg + seed |
  | story_building     |    53 | Gutenberg + seed |
  | tongue_twister     |    15 | curated          |
  | rapid_fire_qa      |    15 | curated          |
  | silly_topic_debate |    10 | curated          |
  | roleplay           |    10 | curated          |
  | yes_and_improv     |     7 | curated          |
  | short_speech       |     6 | curated          |
  | no_but_improv      |     6 | curated          |
  | audio_postcard     |     6 | curated          |
  | TOTAL              |   278 |                  |

Noisy Gutenberg entries filtered at M10 via keyword list (NOISY_KEYWORDS in prep_dataset.py).

## Fine-tuning Dataset (M10)
- Format: ChatML (system / user / assistant triples)
- Input side: exact prompt from engine/prompts/templates.py
- Output side: validated JSON lesson content (passed quality gate)
- Target: 1000 examples total, ~90 per lesson type
- Generation: Gemini Flash + quality gate (TA >= 0.75, no duplicate)
- Pass 1: raw seeds as topic seeds / book_excerpt context
- Pass 2: synthetic topic pool per type to fill remaining quota
- Split: 85% train, 15% eval
- Progress: saved to finetune/data/progress.json (resumable)
- Output: finetune/data/train.jsonl, finetune/data/eval.jsonl
- Estimated cost: ~$0.22 total

## Hardware
- Laptop: RTX 4060 4GB VRAM
  - Inference: fine, runs Llama 3.1 8B via Ollama
  - Fine-tuning: Phi-3 mini 3.8B only (Unsloth QLoRA ~3.5GB VRAM)
  - Mistral 7B / Llama 3.1 8B fine-tuning → OOM risk, defer to DGX
- DGX Spark: Mistral 7B, Llama 3.1 8B fine-tuning (later)
- RTX 4060 Laptop GPU: 8.6GB VRAM (full), not 4GB as previously noted
- Mistral 7B QLoRA now feasible on laptop (needs ~6GB)
- Phi-3 mini 3.8B still primary target (faster, safer margin)
- Llama 3.1 8B QLoRA borderline (~7-8GB) — worth trying

## UI (M8)
- Purpose: demo showcase for CEO + engineers, not a teacher dashboard
- Single HTML/CSS/JS page, no framework
- Engine A vs Engine B selectors: Gemini, DeepSeek, Llama 3.1 8B, Mistral 7B, Llama 3.2 3B
- CEO layer: cost/lesson, TA score, uniqueness — always visible
- Engineer layer: model, token counts, gen time, embedding hash, dominant metrics — collapsible
- Stats bar: cost comparison with % cheaper callout (skipped when one engine is local)
- Local models show actual TA scoring cost, not $0.00
- Color coded per engine: Gemini=teal, DeepSeek=coral, Llama 3.1=purple, Mistral=amber, Llama 3.2=blue

## Benchmark Target
Beat or match existing expensive model on P/Pr/R/Ff/V/TA at <5% of cost.
Judge: Bantrly's own scoring system.
Demo centrepiece: Engine A vs Engine B comparison including fine-tuned Phi-3 mini
showing cost/lesson, TA score, quality, uniqueness, generation time.

## Files Changed
| Date       | Module | Files                                                                                              | Notes                                                                                     |
|------------|---------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 2026-05-28 | M0      | decisions.md, config/rubric.py, config/settings.py, pyproject.toml                                | foundation, uv + Python 3.11                                                              |
| 2026-05-28 | M1      | schema/lesson_types.py                                                                             | 11 typed Pydantic schemas + enums                                                         |
| 2026-05-28 | M2      | scraper/sources.py, scraper/pipeline.py, scraper/cleaner.py                                        | Gutenberg + curated seeds, 278 items                                                      |
| 2026-05-28 | M3      | engine/prompts/base.py, engine/prompts/templates.py                                                | 11 rubric-native prompt builders                                                          |
| 2026-05-28 | M4      | engine/clients/base.py, engine/clients/gemini.py, engine/clients/deepseek.py, config/settings.py  | model clients, cost_usd helper, google-genai SDK                                          |
| 2026-05-28 | M5      | engine/quality_gate.py, vectorstore/dedup.py                                                       | TA via LLM + FAISS dedup + type-aware sanity check                                        |
| 2026-05-28 | M6      | engine/generator.py                                                                                | orchestrator: prompt → generate → gate → retry → ChalkLesson                             |
| 2026-05-28 | M7      | api/main.py                                                                                        | FastAPI: /health, /generate, /generate/batch, /lesson-types                               |
| 2026-05-28 | M8      | ui/index.html, engine/clients/ollama.py, engine/generator.py, api/main.py                         | Engine A/B selector UI, Ollama client, local model support                                |
| 2026-05-28 | M10     | finetune/prep_dataset.py, finetune/data/                                                           | ChatML dataset generation, 1000 examples, resumable pipeline                             |
| 2026-05-29 | M10 | finetune/data/train.jsonl, finetune/data/eval.jsonl, finetune/data/examples.jsonl | 993 examples total (844 train, 149 eval), 85/15 split, all 11 lesson types, ChatML format |

| 2026-05-30 | M11 | finetune/train.py, finetune/output/adapter/ | Phi-3 mini 3.8B QLoRA fine-tuned on 993 ChatML examples. 3 epochs, bf16, LoRA r=16. 115MB adapter. Trained in WSL2 on RTX 4060 8.6GB. Loss ~0.60 end of training |
| 2026-05-30 | M11 | finetune/serve.py | Fine-tuned model inference server. Loads adapter, serves /v1/chat/completions (OpenAI-compatible), port 8001. Run in WSL2. |
| 2026-05-30 | M11 | engine/clients/chalk_phi3.py | Client for fine-tuned Phi-3. Calls localhost:8001. Returns GenerationResult with cost_usd=0.0. |
| 2026-05-30 | M11 | finetune/export_gguf.py, finetune/output/gguf/ | GGUF Q8_0 conversion of adapter. Output in finetune/output/gguf/. |
| 2026-05-30 | M11 | finetune/compare.py | Standalone script: FT Phi-3 vs Gemini vs DeepSeek side-by-side on same prompt. Prints TA score, cost, content. |
| 2026-05-30 | M11 | api/main.py | Added chalk-phi3 model selector. /generate now accepts model=chalk-phi3 in addition to gemini, deepseek, ollama/*. |

## Resolved Decisions
- [x] Gemini API key added to .env
- [x] DeepSeek API key + payment added ($2 credit)
- [x] Migrated from google-generativeai to google-genai SDK
- [x] Fine-tuning base model: Phi-3 mini 3.8B (fits 4GB VRAM safely)
- [x] Scrape sources defined and pipeline working
- [x] Heuristic scoring dropped from quality gate — not meaningful without audio
- [x] Synthetic data volume: 1000 examples total (~90/type)
- [x] Training format: ChatML (system/user/assistant triples)
- [x] Fine-tuned model integrated end-to-end: serve.py → chalk_phi3 client → /generate API
- [x] GGUF export complete (Q8_0, finetune/output/gguf/)

## Open Decisions
- [ ] DGX Spark setup timeline (later)
- [ ] DB wiring for lesson persistence (SQLite, defer to after demo)
- [ ] Eval dashboard (M9) — building after M11 fine-tuning so fine-tuned model can be included. eval/ is currently empty.
- [ ] ui/index.html is missing — ui/ directory is empty. M8 UI needs to be rebuilt or restored.

## Observations Summary
See observations.md for full detail. Key findings so far:
- DeepSeek defaults to "She sells seashells" variants — narrow creative range
- Gemini generates original content consistently
- Dedup system correctly catches semantic near-duplicates
- Retry loop inflates cost when model lacks diversity (DeepSeek 4× more expensive on run 2)
- Local models (Llama, Mistral) generate shorter content by default — sanity check needed type-awareness
- TA scoring is the dominant cost for local model pipelines (~$0.000035/call)
- Output verbosity varies by model — more tokens ≠ better quality