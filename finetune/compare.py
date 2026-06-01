import json
import sys
sys.path.insert(0, '/mnt/d/Bantrly/code/lesson-engine')

from unsloth import FastLanguageModel
from engine.prompts.templates import build_prompt
from engine.clients.gemini import generate as gemini_gen
from engine.clients.deepseek import generate as deepseek_gen
from engine.quality_gate import score_ta_llm
from schema.lesson_types import LessonType, AgeGroup, Difficulty, TeacherContext

LESSON_TYPE = LessonType.TONGUE_TWISTER
AGE         = AgeGroup.TEEN
DIFF        = Difficulty.HARD
TOPIC       = "S and SH sounds"

system, user = build_prompt(LESSON_TYPE, TOPIC, AGE, DIFF, TeacherContext())

# ── Fine-tuned Phi-3 ───────────────────────────────────────────────────────
print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="finetune/output/adapter",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

inputs = tokenizer.apply_chat_template(
    [{"role": "system", "content": system},
     {"role": "user",   "content": user}],
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.9,
    do_sample=True,
)
phi_raw     = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
phi_content = json.loads(phi_raw)
phi_ta, _   = score_ta_llm(LESSON_TYPE, AGE, DIFF, TOPIC, phi_content)

# ── Gemini ─────────────────────────────────────────────────────────────────
print("Calling Gemini...")
g    = gemini_gen(system, user)
g_ta = score_ta_llm(LESSON_TYPE, AGE, DIFF, TOPIC, g.content)[0] if g.ok else 0.0

# ── DeepSeek ───────────────────────────────────────────────────────────────
print("Calling DeepSeek...")
d    = deepseek_gen(system, user)
d_ta = score_ta_llm(LESSON_TYPE, AGE, DIFF, TOPIC, d.content)[0] if d.ok else 0.0

# ── Results ────────────────────────────────────────────────────────────────
results = [
    ("Fine-tuned Phi-3 mini", phi_content.get("text",""), phi_ta, 0.0,    "local"),
    ("Gemini 2.5 Flash",      g.content.get("text","") if g.ok else "FAILED", g_ta, g.cost_usd, "api"),
    ("DeepSeek V3",           d.content.get("text","") if d.ok else "FAILED", d_ta, d.cost_usd, "api"),
]

print("\n" + "="*70)
for name, text, ta, cost, kind in results:
    cost_str = "FREE (local)" if kind == "local" else f"${cost:.6f}"
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  TA Score : {ta*100:.0f}%")
    print(f"  Cost     : {cost_str}")
    print(f"  Content  : {text}")
print(f"\n{'='*70}")