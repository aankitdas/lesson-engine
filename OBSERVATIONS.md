# Chalk — Live Observations

## Observation 001 — DeepSeek defaults to classics, Gemini generates original
**Date**: 2026-05-28
**Test**: Tongue Twister, Teen, Hard, "S and SH sounds" — run twice

On both runs, DeepSeek gravitated toward "She sells seashells" variants.
Gemini produced original content each time (Sophisticated Sasha..., Stylish Sheena...).

**Implication**: DeepSeek's training data over-represents classic tongue twisters.
For a lesson engine, this is a content quality problem — students will see the same
material repeatedly. A fine-tuned specialist trained on diverse, curated data
would not have this bias.

---

## Observation 002 — Dedup system caught DeepSeek's repetition
**Date**: 2026-05-28
**Test**: Same as above, second run

DeepSeek's second "She sells seashells" variant was flagged by FAISS (cosine similarity > 0.85),
quality gate failed it, retry loop fired 3 times. All 3 retries were semantically
too close to the first run's output.

**Implication**: Without a dedup system, a production lesson engine would silently
serve near-identical content to students. The semantic dedup layer (not exact match)
is essential — surface-level rephrasing of the same content is still a duplicate.

---

## Observation 003 — Retry loop inflates cost when model lacks diversity
**Date**: 2026-05-28
**Test**: Same as above

DeepSeek cost jumped from $0.000211 (run 1) to $0.000689 (run 2) — 3× increase.
Gemini stayed consistent at ~$0.000171.

Cost breakdown for DeepSeek run 2:
  - 3 generation attempts × ~$0.000211 avg = ~$0.000633
  - 3 TA scoring calls via Gemini = ~$0.000056
  - Total: $0.000689

Gemini ended up 75% cheaper on run 2 — not because of per-token pricing
but because it generated acceptable, unique content on the first attempt.

**Implication**: Raw per-token pricing is misleading. True cost = price × attempts.
A model that consistently generates unique, high-quality content on attempt 1
is cheaper in practice regardless of its headline rate.

---

## Observation 004 — Output verbosity varies by model and lesson type
**Date**: 2026-05-28
**Test**: Story Building, Teen, Easy, "space exploration"

Gemini: ~1073 tokens, $0.000161
DeepSeek: ~1407 tokens, $0.000211

DeepSeek wrote ~31% more tokens for the same task. More verbose ≠ better quality.
Both passed the gate with 100% TA score.

**Implication**: A fine-tuned model can be trained to be concise and on-format,
reducing token usage without sacrificing quality. This is an additional cost
lever beyond per-token pricing.

---
## Observation 005 — Local models generate shorter content by default
**Date**: 2026-05-28
**Test**: Tongue Twister, Teen, Medium, Llama 3.1 8B local

Llama 3.1 8B consistently generated 4-9 word tongue twisters at medium difficulty.
Sanity check min_words=10 was rejecting all 3 attempts, so TA scorer never ran → TA=0.0.
Content quality was fine — short tongue twisters are valid (e.g. "Red lorry, yellow lorry").

Fix: type-aware minimum word counts in sanity_check.
Tongue twister min lowered to 4 words.

**Implication**: Local models tend toward brevity. For fine-tuning, training examples
should include a mix of short and long content so the model learns appropriate
length calibration per lesson type.

## Observation 006 — Generation cost for local models is just the TA scoring call
**Date**: 2026-05-28

Llama 3.1 8B generation cost: $0.000035 — this is entirely the Gemini TA scorer call.
The generation itself is $0.00. At scale, the TA scoring becomes the dominant cost
for local model pipelines. Future optimisation: batch TA scoring or use a local
scorer to eliminate this cost entirely.

---
## Observation 007 — DeepSeek still defaulting to "She sells seashells"
**Date**: 2026-05-28
**Test**: Tongue Twister, S and SH sounds — multiple runs

DeepSeek has now produced "She sells seashells" variants in 3 out of 4 runs.
Gemini produces original content every single time.
DeepSeek passed this time only because the exact embedding was different enough
from previous runs to clear the 0.85 similarity threshold — but it's clearly
drawing from the same distribution.

**Implication**: DeepSeek's instruction-following is fine but its creative range
for this task is narrow. This is exactly the problem fine-tuning solves —
training on diverse, curated examples breaks the model out of high-frequency
training data patterns.
---
## Observation 008 — Dataset generation took 4+ hours sequentially
**Date**: 2026-05-29
**Fix needed**: async/parallel generation for future runs
993 examples × ~2.5s avg = ~41 minutes theoretical minimum
Actual: 4+ hours due to sequential execution + sleep delays + multiple crashes/restarts
Next run: rewrite with ThreadPoolExecutor (15 workers) → estimated ~5-10 minutes

---

## Observation 009 — Fine-tuned Phi-3 mini generates valid JSON on first attempt
**Date**: 2026-05-30
**Test**: Tongue Twister, Teen, Hard, S and SH sounds

Fine-tuned model produced valid JSON, correct schema, S/SH dense content,
proper difficulty note — on first attempt, no quality gate needed.
Content quality subjectively comparable to Gemini Flash output.
Next step: formal comparison via TA scorer across all 11 lesson types.

---
## Observation 010 — Fine-tuned Phi-3 mini matches Gemini, beats DeepSeek
**Date**: 2026-05-30
**Test**: Tongue Twister, Teen, Hard, S and SH sounds

| Model              | TA Score | Cost      | Content quality     |
|--------------------|----------|-----------|---------------------|
| Fine-tuned Phi-3   | 100%     | FREE      | Original, creative  |
| Gemini 2.5 Flash   | 100%     | $0.000127 | Original, creative  |
| DeepSeek V3        | 100%     | $0.000207 | "She sells seashells" again |

Fine-tuned Phi-3 mini achieves identical TA score to Gemini at zero 
inference cost. DeepSeek scores equal TA but produces generic content
for the 4th+ time on this topic — quality gate passes it but dedup
would catch it in production.

Core thesis demonstrated: task-specific fine-tuned small model ≥ 
general large model for this narrow task.
---
## Observation 011 — Local inference speed vs cloud API
**Date**: 2026-05-31

Chalk Phi-3 (RTX 4060 local): 37.4s, ~8 tokens/second
DeepSeek V3 (cloud):          12.6s, ~100+ tokens/second

Speed gap is hardware, not model quality. Production deployment on 
A100/H100 would reduce Chalk Phi-3 inference to 2-3 seconds at 
near-zero cost vs $0.000498 per lesson for DeepSeek.
RTX 4060 laptop is proof-of-concept only.
---
## Hypotheses to test with fine-tuned model
- [x] Fine-tuned Phi-3 mini will not default to "She sells seashells" variants — confirmed Obs 009/010
- [x] Fine-tuned model will match TA scores of Gemini Flash at fraction of cost — confirmed Obs 010 (100% TA, $0.00)
- [ ] Fine-tuned model will use fewer tokens (concise, on-format by training)
- [ ] Dedup rejection rate will be lower for fine-tuned model vs base DeepSeek