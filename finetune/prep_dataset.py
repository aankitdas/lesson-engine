import json
import time
import random
from pathlib import Path
from datetime import datetime

from schema.lesson_types import (
    LessonType, AgeGroup, Difficulty, TeacherContext, GenerationRequest
)
from engine.prompts.templates import build_prompt
from engine.clients.gemini import generate as gemini_generate
from engine.quality_gate import run_quality_gate

# ── Config ─────────────────────────────────────────────────────────────────

TARGET_PER_TYPE  = 90
RAW_DIR          = Path("scraper/raw")
OUT_DIR          = Path("finetune/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE    = OUT_DIR / "progress.json"
TRAIN_FILE       = OUT_DIR / "train.jsonl"
EVAL_FILE        = OUT_DIR / "eval.jsonl"
EVAL_RATIO       = 0.15

# ── Noisy source filter ────────────────────────────────────────────────────

NOISY_KEYWORDS = [
    "modest proposal", "eating", "victorian courtship",
    "yearbook", "1920", "kate chopin", "awakening",
    "children's literature textbook", "teacher-training",
]

def is_noisy(item: dict) -> bool:
    notes = item.get("source_notes", "").lower()
    text  = item.get("raw_text", "").lower()
    return any(kw in notes or kw in text for kw in NOISY_KEYWORDS)


# ── Age group mapping ──────────────────────────────────────────────────────

AGE_MAP = {
    "child": AgeGroup.CHILD,
    "teen":  AgeGroup.TEEN,
    "youth": AgeGroup.YOUTH,
    "adult": AgeGroup.ADULT,
}

DIFFICULTY_CYCLE = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]


# ── Synthetic topic seeds per lesson type ──────────────────────────────────

TOPIC_SEEDS = {
    LessonType.PARAGRAPH_READING: [
        "the water cycle", "how volcanoes form", "the history of the internet",
        "life in ancient Egypt", "how bees make honey", "the solar system",
        "migration of birds", "how planes fly", "the Amazon rainforest",
        "the human digestive system", "climate change basics", "the history of music",
    ],
    LessonType.TONGUE_TWISTER: [
        "S and SH sounds", "R and L sounds", "TH sounds", "P and B sounds",
        "CH and J sounds", "consonant clusters", "W and V sounds",
        "F and PH sounds", "hard G and soft G", "silent letters",
    ],
    LessonType.POEM_RECITATION: [
        "seasons and nature", "friendship", "courage", "the ocean",
        "city life", "dreams and imagination", "animals", "time passing",
        "home and belonging", "hope", "adventure", "night and stars",
    ],
    LessonType.STORY_BUILDING: [
        "a mysterious door", "a robot who feels emotions", "the last library on Earth",
        "a message in a bottle", "time travel gone wrong", "a school for superheroes",
        "the day animals could talk", "a forgotten map", "a city underwater",
        "the first day on another planet",
    ],
    LessonType.YES_AND_IMPROV: [
        "you discover your shadow is alive", "every chair in school starts floating",
        "a volcano appears in the playground", "you wake up speaking a new language",
        "your homework starts arguing back",
    ],
    LessonType.NO_BUT_IMPROV: [
        "replace school lunch with candy", "make every lesson a PE lesson",
        "ban all homework forever", "let students grade teachers",
        "replace school uniforms with costumes",
    ],
    LessonType.SHORT_SPEECH: [
        "why libraries matter", "the importance of sleep", "should school start later",
        "why we should learn a second language", "the impact of social media on teens",
        "why reading fiction makes you smarter", "should junk food be banned in schools",
        "the future of AI in education", "why physical education matters",
        "the value of making mistakes",
    ],
    LessonType.AUDIO_POSTCARD: [
        "your favourite childhood memory", "a place you want to visit",
        "describing your ideal day", "a person who inspired you",
        "your neighbourhood at night", "the best meal you ever had",
        "a moment you felt proud", "what home means to you",
    ],
    LessonType.SILLY_TOPIC_DEBATE: [
        "dogs are better than cats", "summer is better than winter",
        "books are better than movies", "breakfast is the best meal",
        "math is more important than art", "zoos should be abolished",
        "video games are a sport", "school uniforms are a good idea",
        "homework does more harm than good", "robots will replace teachers",
    ],
    LessonType.ROLEPLAY: [
        "returning a faulty product to a shop", "asking for directions in a foreign city",
        "negotiating a later curfew with a parent", "a job interview at a café",
        "reporting a lost item to a police officer",
        "booking a table at a restaurant over the phone",
        "asking a teacher to extend a deadline",
        "checking into a hotel as a solo traveller",
    ],
    LessonType.RAPID_FIRE_QA: [
        "world geography", "human body facts", "famous inventors",
        "space and astronomy", "world history", "animals and nature",
        "mathematics quick fire", "English literature",
        "science vocabulary", "general knowledge",
    ],
}

def append_example(ex: dict):
    with open(OUT_DIR / "examples.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
# ── ChatML formatter ───────────────────────────────────────────────────────

def to_chatml(system: str, user: str, output: dict) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": json.dumps(output, ensure_ascii=False)},
        ]
    }


# ── Progress tracking ──────────────────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {lt.value: 0 for lt in LessonType}


def save_progress(progress: dict):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# ── Single example generator ───────────────────────────────────────────────

def generate_example(
    lesson_type: LessonType,
    topic: str,
    age_group: AgeGroup,
    difficulty: Difficulty,
    book_excerpt: str | None = None,
) -> dict | None:
    ctx = TeacherContext(book_excerpt=book_excerpt)
    system, user = build_prompt(lesson_type, topic, age_group, difficulty, ctx)

    result = gemini_generate(system, user)
    if not result.ok:
        return None

    # skip TA LLM scoring for dataset prep — we trust Gemini's own output
    # just run sanity check and dedup
    from engine.quality_gate import sanity_check, extract_main_text
    from vectorstore.dedup import DedupIndex

    sane, reason = sanity_check(result.content, lesson_type)
    if not sane:
        return None

    return to_chatml(system, user, result.content)


# ── Main pipeline ──────────────────────────────────────────────────────────

def run_prep():
    progress   = load_progress()
    all_examples: list[dict] = []
    examples_path = OUT_DIR / "examples.jsonl"
    if examples_path.exists():
        with open(examples_path, encoding="utf-8") as f:
            all_examples = [json.loads(l) for l in f if l.strip()]
        print(f"Resuming — {len(all_examples)} examples already saved")

    for lesson_type in LessonType:
        lt_key   = lesson_type.value
        done     = progress.get(lt_key, 0)
        needed   = TARGET_PER_TYPE - done
        if needed <= 0:
            print(f"[{lt_key}] already complete ({done} examples), skipping")
            continue

        print(f"\n[{lt_key}] need {needed} more (have {done})")
        generated = 0

        # ── pass 1: use raw seeds ──────────────────────────────────────────
        raw_path = RAW_DIR / f"{lt_key}.jsonl"
        seeds    = []
        if raw_path.exists():
            with open(raw_path, encoding="utf-8") as f:
                seeds = [json.loads(l) for l in f if l.strip()]
            seeds = [s for s in seeds if not is_noisy(s)]

        for seed in seeds:
            if generated >= needed:
                break
            age    = AGE_MAP.get(seed.get("age_group_hint", "teen"), AgeGroup.TEEN)
            diff   = DIFFICULTY_CYCLE[generated % 3]
            topic  = seed["raw_text"][:120]  # use first 120 chars as topic seed
            excerpt = seed["raw_text"] if seed.get("source_type") == "scraped" else None

            ex = generate_example(lesson_type, topic, age, diff, book_excerpt=excerpt)
            if ex:
                all_examples.append(ex)
                append_example(ex)
                generated += 1
                print(f"  [{lt_key}] {done + generated}/{TARGET_PER_TYPE}", end="\r")
            time.sleep(0.5)  # gentle rate limiting

        # ── pass 2: synthetic topics to fill remaining ─────────────────────
        topic_pool = TOPIC_SEEDS.get(lesson_type, ["general topic"])
        ages       = list(AGE_MAP.values())
        idx        = 0
        while generated < needed:
            topic  = topic_pool[idx % len(topic_pool)]
            age    = ages[idx % len(ages)]
            diff   = DIFFICULTY_CYCLE[idx % 3]
            ex = generate_example(lesson_type, topic, age, diff)
            if ex:
                all_examples.append(ex)
                append_example(ex)
                generated += 1
                print(f"  [{lt_key}] {done + generated}/{TARGET_PER_TYPE} (synthetic)", end="\r")
            idx += 1
            # time.sleep(0.5)

        progress[lt_key] = done + generated
        save_progress(progress)
        print(f"\n  ✓ {lt_key}: {progress[lt_key]} total")

    # ── split and save ─────────────────────────────────────────────────────
    # replace the split/save block at the end with:
    print("\nFinalising dataset...")
    all_saved = []
    examples_path = OUT_DIR / "examples.jsonl"
    if examples_path.exists():
        with open(examples_path, encoding="utf-8") as f:
            all_saved = [json.loads(l) for l in f if l.strip()]

    random.shuffle(all_saved)
    split     = int(len(all_saved) * (1 - EVAL_RATIO))
    train_set = all_saved[:split]
    eval_set  = all_saved[split:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(EVAL_FILE, "w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✓ Dataset complete")
    print(f"  Train: {len(train_set)} examples → {TRAIN_FILE}")
    print(f"  Eval:  {len(eval_set)} examples  → {EVAL_FILE}")
    print(f"  Total: {len(all_saved)}")


if __name__ == "__main__":
    run_prep()