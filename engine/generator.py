import hashlib
from config.rubric import DOMINANT_METRICS, CHALK_MAX_RETRIES
from schema.lesson_types import (
    LessonType, AgeGroup, Difficulty, TeacherContext,
    ChalkLesson, ChalkMeta, GenerationRequest, ContentType,
    ParagraphReadingContent, TongueTwisterContent, PoemRecitationContent,
    StoryBuildingContent, YesAndImprovContent, NoBuImprovContent,
    ShortSpeechContent, AudioPostcardContent,
    SillyTopicDebateContent, RoleplayContent, RapidFireQAContent,
)
from engine.prompts.templates import build_prompt
from engine.clients.gemini import generate as gemini_generate
from engine.clients.deepseek import generate as deepseek_generate
from engine.quality_gate import run_quality_gate
from engine.clients.ollama import generate as ollama_generate
from engine.clients.chalk_phi3 import generate as chalk_generate

_CONTENT_MODELS = {
    LessonType.PARAGRAPH_READING:  ParagraphReadingContent,
    LessonType.TONGUE_TWISTER:     TongueTwisterContent,
    LessonType.POEM_RECITATION:    PoemRecitationContent,
    LessonType.STORY_BUILDING:     StoryBuildingContent,
    LessonType.YES_AND_IMPROV:     YesAndImprovContent,
    LessonType.NO_BUT_IMPROV:      NoBuImprovContent,
    LessonType.SHORT_SPEECH:       ShortSpeechContent,
    LessonType.AUDIO_POSTCARD:     AudioPostcardContent,
    LessonType.SILLY_TOPIC_DEBATE: SillyTopicDebateContent,
    LessonType.ROLEPLAY:           RoleplayContent,
    LessonType.RAPID_FIRE_QA:      RapidFireQAContent,
}


def _parse_content(lesson_type: LessonType, content_dict: dict) -> ContentType:
    model_class = _CONTENT_MODELS[lesson_type]
    valid_fields = model_class.model_fields.keys()
    return model_class(**{k: v for k, v in content_dict.items() if k in valid_fields})


def _text_hash(content_dict: dict) -> str:
    return hashlib.sha256(str(content_dict).encode()).hexdigest()[:16]

def _call_model(model: str, system: str, user: str):
    if model == "deepseek":
        return deepseek_generate(system, user)
    if model == "chalk-phi3":
        return chalk_generate(system, user)
    if model.startswith("ollama/"):
        return ollama_generate(system, user, model=model.split("/", 1)[1])
    return gemini_generate(system, user)


def generate_lesson(request: GenerationRequest, model: str = "gemini") -> ChalkLesson:
    system, user = build_prompt(
        request.lesson_type,
        request.topic,
        request.age_group,
        request.difficulty,
        request.teacher_context,
    )

    best_result = None
    best_gate   = None
    total_cost  = 0.0

    for attempt in range(1, CHALK_MAX_RETRIES + 1):
        gen = _call_model(model, system, user)
        total_cost += gen.cost_usd

        if not gen.ok:
            print(f"  [attempt {attempt}] generation error — {gen.error}")
            continue

        gate = run_quality_gate(
            request.lesson_type,
            request.age_group,
            request.difficulty,
            request.topic,
            gen.content,
        )
        total_cost += gate["ta_cost_usd"]

        best_result = gen
        best_gate   = gate

        if gate["passed"]:
            break

        print(f"  [attempt {attempt}] failed gate — {gate['fail_reason']}")

    if best_result is None:
        raise RuntimeError(f"Generation failed after {CHALK_MAX_RETRIES} attempts")

    try:
        content = _parse_content(request.lesson_type, best_result.content)
    except Exception as e:
        raise RuntimeError(f"Content parsing failed: {e} | raw: {best_result.content}")

    meta = ChalkMeta(
        lesson_type=request.lesson_type,
        age_group=request.age_group,
        difficulty=request.difficulty,
        topic=request.topic,
        teacher_tags=request.teacher_context.buzzwords,
        generation_model=best_result.model,
        generation_cost_usd=round(total_cost, 6),
        dominant_metrics=DOMINANT_METRICS.get(request.lesson_type.value, []),
        rubric_scores={
            "TA": best_gate["ta_score"],
            "input_tokens": best_result.input_tokens,
            "output_tokens": best_result.output_tokens,
        },
        weighted_score=best_gate["ta_score"],
        embedding_hash=_text_hash(best_result.content),
        is_duplicate=best_gate["is_duplicate"],
        passed_quality_gate=best_gate["passed"],
    )

    return ChalkLesson(meta=meta, content=content)


def generate_batch(request: GenerationRequest, model: str = "gemini") -> list[ChalkLesson]:
    lessons = []
    for i in range(request.batch_size):
        print(f"  Generating {i + 1}/{request.batch_size}...")
        try:
            lessons.append(generate_lesson(request, model=model))
        except RuntimeError as e:
            print(f"  [SKIP] {e}")
    return lessons