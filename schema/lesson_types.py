from pydantic import BaseModel, Field
from typing import Optional, Union
from enum import Enum
import uuid
from datetime import datetime


# ── Enums ──────────────────────────────────────────────────────────────────

class LessonType(str, Enum):
    PARAGRAPH_READING   = "paragraph_reading"
    TONGUE_TWISTER      = "tongue_twister"
    POEM_RECITATION     = "poem_recitation"
    STORY_BUILDING      = "story_building"
    YES_AND_IMPROV      = "yes_and_improv"
    NO_BUT_IMPROV       = "no_but_improv"
    SHORT_SPEECH        = "short_speech"
    AUDIO_POSTCARD      = "audio_postcard"
    SILLY_TOPIC_DEBATE  = "silly_topic_debate"
    ROLEPLAY            = "roleplay"
    RAPID_FIRE_QA       = "rapid_fire_qa"

class AgeGroup(str, Enum):
    CHILD  = "child"   # ≤12
    TEEN   = "teen"    # 13-15
    YOUTH  = "youth"   # 16-18
    ADULT  = "adult"   # 19+

class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


# ── Chalk metadata (attached to every lesson) ──────────────────────────────

class ChalkMeta(BaseModel):
    lesson_id:          str     = Field(default_factory=lambda: str(uuid.uuid4()))
    lesson_type:        LessonType
    age_group:          AgeGroup
    difficulty:         Difficulty
    topic:              str
    teacher_tags:       list[str]               = []
    generation_model:   str                     = ""
    generation_cost_usd: float                  = 0.0
    dominant_metrics:   list[str]               = []
    rubric_scores:      dict[str, float]        = {}
    weighted_score:     float                   = 0.0
    embedding_hash:     str                     = ""
    is_duplicate:       bool                    = False
    passed_quality_gate: bool                   = False
    created_at:         datetime                = Field(default_factory=datetime.utcnow)


# ── Teacher optional input ─────────────────────────────────────────────────

class TeacherContext(BaseModel):
    buzzwords:      list[str]       = []
    event_note:     Optional[str]   = None
    book_excerpt:   Optional[str]   = None
    extra_notes:    Optional[str]   = None


# ── Non-interactive lesson content schemas ─────────────────────────────────

class ParagraphReadingContent(BaseModel):
    passage:            str
    word_count:         int
    target_phonemes:    list[str]   = []    # e.g. ["th", "str", "r"]
    difficulty_cues:    list[str]   = []    # e.g. ["long sentences", "complex vocab"]
    estimated_duration_seconds: int = 60

class TongueTwisterContent(BaseModel):
    text:                 Optional[str] = None
    target_phoneme:       Optional[str] = None
    repetitions_required: int           = 3
    difficulty_note:      Optional[str] = None

class PoemRecitationContent(BaseModel):
    title:           Optional[str] = None
    text:            Optional[str] = None
    stress_pattern:  Optional[str] = None
    emotional_tone:  Optional[str] = None
    prosody_notes:   Optional[str] = None

class StoryBuildingContent(BaseModel):
    opening_prompt:     str
    continuation_cues:  list[str]       = []    # 2-3 prompts to keep story going
    genre:              Optional[str]   = None
    min_turns:          int             = 3

class YesAndImprovContent(BaseModel):
    scenario_seed:      str
    escalation_hooks:   list[str]       = []
    tone:               Optional[str]   = "playful"

class NoBuImprovContent(BaseModel):
    scenario_seed:      str
    escalation_hooks:   list[str]       = []
    tone:               Optional[str]   = "playful"

class ShortSpeechContent(BaseModel):
    topic:              str
    outline:            list[str]       = []    # e.g. ["intro", "point1", "point2", "conclusion"]
    target_duration_seconds: int        = 90
    constraints:        Optional[str]   = None  # e.g. "no filler words"

class AudioPostcardContent(BaseModel):
    prompt:             str
    context:            Optional[str]   = None
    suggested_elements: list[str]       = []    # e.g. ["greeting", "description", "feeling", "closing"]


# ── Interactive lesson content schemas ─────────────────────────────────────

class SillyTopicDebateContent(BaseModel):
    topic:              str
    student_stance:     str                     # "for" or "against"
    ai_lines:           list[str]               # AI's side of the debate
    debate_hooks:       list[str]       = []    # provocations/counterarguments

class RoleplayContent(BaseModel):
    scenario:           str
    student_role:       str
    ai_role:            str
    ai_lines:           list[str]
    conversation_goal:  Optional[str]   = None

class RapidFireQAContent(BaseModel):
    topic:              str
    questions:          list[str]
    ai_lines:           list[str]               # AI prompts between answers
    expected_answer_length: str         = "short"   # short / medium


# ── Unified lesson wrapper ─────────────────────────────────────────────────


ContentType = Union[
    ParagraphReadingContent,  TongueTwisterContent,   PoemRecitationContent,
    StoryBuildingContent,     YesAndImprovContent,     NoBuImprovContent,
    ShortSpeechContent,       AudioPostcardContent,    SillyTopicDebateContent,
    RoleplayContent,          RapidFireQAContent
]

class ChalkLesson(BaseModel):
    meta:       ChalkMeta
    content:    ContentType


# ── Generation request (teacher → engine) ─────────────────────────────────

class GenerationRequest(BaseModel):
    lesson_type:        LessonType
    age_group:          AgeGroup
    difficulty:         Difficulty
    topic:              str
    teacher_context:    TeacherContext  = TeacherContext()
    batch_size:         int             = Field(default=1, ge=1, le=10)