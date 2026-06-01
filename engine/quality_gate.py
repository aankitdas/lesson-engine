import json
import re
from config.settings import GEMINI_MODEL, cost_usd, GEMINI_API_KEY
from config.rubric import CHALK_PASS_THRESHOLD
from schema.lesson_types import LessonType, AgeGroup, Difficulty
from vectorstore.dedup import DedupIndex

_dedup = DedupIndex()


def extract_main_text(content: dict) -> str:
    for key in ("passage", "text", "opening_prompt", "scenario_seed", "prompt", "scenario"):
        if key in content and isinstance(content[key], str):
            if len(content[key].split()) >= 8:
                return content[key]
    for key in ("questions", "ai_lines", "outline", "continuation_cues", "escalation_hooks"):
        if key in content and isinstance(content[key], list) and content[key]:
            return " ".join(str(item) for item in content[key])
    # fallback — join all strings, prefer longer ones
    strings = [(k, v) for k, v in content.items() if isinstance(v, str)]
    strings.sort(key=lambda x: len(x[1]), reverse=True)
    return strings[0][1] if strings else ""


MIN_WORDS = {
    "tongue_twister":  4,
    "yes_and_improv":  5,
    "no_but_improv":   5,
    "audio_postcard":  5,
    "rapid_fire_qa":   8,
}

def sanity_check(content: dict, lesson_type: LessonType | None = None) -> tuple[bool, str]:
    if not content:
        return False, "empty content"
    text = extract_main_text(content)
    words = len(text.split())
    lt_key = lesson_type.value if lesson_type else ""
    min_w = MIN_WORDS.get(lt_key, 10)
    if words < min_w:
        return False, f"too short ({words} words)"
    if words > 800:
        return False, f"too long ({words} words)"
    return True, ""


def score_ta_llm(
    lesson_type: LessonType,
    age_group: AgeGroup,
    difficulty: Difficulty,
    topic: str,
    content: dict,
) -> tuple[float, float]:
    from google import genai
    from google.genai import types

    system = (
        "You are a strict lesson quality evaluator for a K-12 spoken English app. "
        "Score Task Adherence (TA) from 0.0 to 1.0. "
        "Respond ONLY with valid JSON: {\"TA\": <float>}"
    )
    user = f"""Lesson type: {lesson_type.value}
Age group: {age_group.value}
Difficulty: {difficulty.value}
Topic: {topic}
Generated content:
{json.dumps(content, indent=2)}

Score TA (0.0–1.0):
- Does the format exactly match the lesson type?
- Is vocabulary and complexity right for the age group?
- Does the content address the stated topic?
- Are difficulty constraints followed?"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )
        data = json.loads(response.text)
        ta = float(data.get("TA", 0.5))
        usage = response.usage_metadata
        c = cost_usd(GEMINI_MODEL,
                     usage.prompt_token_count or 0,
                     usage.candidates_token_count or 0)
        return round(ta, 4), c
    except Exception:
        return 0.5, 0.0


def run_quality_gate(
    lesson_type: LessonType,
    age_group: AgeGroup,
    difficulty: Difficulty,
    topic: str,
    content: dict,
) -> dict:
    # 1. sanity check
    sane, reason = sanity_check(content, lesson_type)
    if not sane:
        return {
            "passed": False, "ta_score": 0.0,
            "is_duplicate": False, "similarity": 0.0,
            "fail_reason": reason, "ta_cost_usd": 0.0,
        }

    # 2. TA via LLM
    ta, ta_cost = score_ta_llm(lesson_type, age_group, difficulty, topic, content)

    # 3. dedup
    text = extract_main_text(content)
    is_dup, similarity = _dedup.is_duplicate(text)

    passed = ta >= CHALK_PASS_THRESHOLD and not is_dup

    if passed:
        _dedup.add(text)

    return {
        "passed":       passed,
        "ta_score":     ta,
        "is_duplicate": is_dup,
        "similarity":   similarity,
        "fail_reason":  "duplicate" if is_dup else ("low TA" if ta < CHALK_PASS_THRESHOLD else ""),
        "ta_cost_usd":  ta_cost,
    }