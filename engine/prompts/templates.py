from schema.lesson_types import (
    AgeGroup, Difficulty, LessonType, TeacherContext,
    ParagraphReadingContent, TongueTwisterContent, PoemRecitationContent,
    StoryBuildingContent, YesAndImprovContent, NoBuImprovContent,
    ShortSpeechContent, AudioPostcardContent,
    SillyTopicDebateContent, RoleplayContent, RapidFireQAContent,
)
from engine.prompts.base import base_system, build_teacher_context, DIFFICULTY_GUIDANCE


def _ctx(teacher_context: TeacherContext) -> str:
    c = build_teacher_context(teacher_context)
    return f"\nTeacher context:\n{c}" if c else ""


# ── Non-interactive ────────────────────────────────────────────────────────

def paragraph_reading(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                      teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.PARAGRAPH_READING, age_group)
    user = f"""Generate a paragraph reading lesson.
Topic: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Return JSON:
{{
  "passage": "<2-4 paragraphs, 80-180 words, designed for reading aloud>",
  "word_count": <integer>,
  "target_phonemes": ["<phoneme1>", "<phoneme2>"],
  "difficulty_cues": ["<what makes this passage challenging to speak>"],
  "estimated_duration_seconds": <integer>
}}"""
    return system, user


def tongue_twister(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                   teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.TONGUE_TWISTER, age_group)
    user = f"""Generate a tongue twister lesson.
Target phoneme/sound: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Rules:
- Must be genuinely difficult to say quickly
- Should be memorable and rhythmic
- 1-3 sentences max

Return JSON:
{{
  "text": "<the tongue twister>",
  "target_phoneme": "<primary phoneme cluster>",
  "repetitions_required": 3,
  "difficulty_note": "<max 1 sentence: what makes this hard to say>"
}}"""
    return system, user


def poem_recitation(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                    teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.POEM_RECITATION, age_group)
    user = f"""Generate an original short poem for recitation practice.
Theme: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Rules:
- 2-4 stanzas, 4 lines each
- Must have clear stress pattern and natural rhythm when spoken aloud
- Avoid forced rhymes that distort pronunciation

Return JSON:
{{
  "title": "<poem title>",
  "text": "<full poem with line breaks as \\n>",
  "stress_pattern": "<e.g. iambic, trochaic, free verse>",
  "emotional_tone": "<e.g. playful, reflective, inspiring>",
  "prosody_notes": "<1 sentence: how to perform it>"
}}"""
    return system, user


def story_building(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                   teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.STORY_BUILDING, age_group)
    user = f"""Generate a story building prompt for a spoken English lesson.
Genre/theme: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Return JSON:
{{
  "opening_prompt": "<1-2 sentences that start a story and demand continuation>",
  "continuation_cues": ["<cue 1>", "<cue 2>", "<cue 3>"],
  "genre": "<genre>",
  "min_turns": 3
}}"""
    return system, user


def yes_and_improv(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                   teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.YES_AND_IMPROV, age_group)
    user = f"""Generate a Yes-And improv scenario for spoken English practice.
Theme: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Rules for Yes-And: student must accept the premise and add to it each turn.
Make the scenario absurd enough to spark creativity but clear enough to follow.

Return JSON:
{{
  "scenario_seed": "<opening scenario, 1-2 sentences>",
  "escalation_hooks": ["<hook 1 to keep it going>", "<hook 2>", "<hook 3>"],
  "tone": "playful"
}}"""
    return system, user


def no_but_improv(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                  teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.NO_BUT_IMPROV, age_group)
    user = f"""Generate a No-But improv scenario for spoken English practice.
Theme: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Rules for No-But: student must disagree with each proposal and offer an alternative.
Make proposals realistic enough that disagreeing requires real reasoning.

Return JSON:
{{
  "scenario_seed": "<a proposal or situation to disagree with, 1-2 sentences>",
  "escalation_hooks": ["<follow-up proposal 1>", "<follow-up proposal 2>", "<follow-up proposal 3>"],
  "tone": "playful"
}}"""
    return system, user


def short_speech(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                 teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.SHORT_SPEECH, age_group)
    user = f"""Generate a short informative speech topic and structure.
Topic: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Return JSON:
{{
  "topic": "<specific speech title>",
  "outline": ["<intro hook>", "<main point 1>", "<main point 2>", "<conclusion/call to action>"],
  "target_duration_seconds": 90,
  "constraints": "<1 sentence: specific delivery rule e.g. no filler words, maintain eye contact>"
}}"""
    return system, user


def audio_postcard(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                   teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.AUDIO_POSTCARD, age_group)
    user = f"""Generate an audio postcard prompt for a spoken English lesson.
Theme: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

An audio postcard is a short personal spoken message (30-60 seconds) from one person to another.

Return JSON:
{{
  "prompt": "<clear scenario: who is speaking, to whom, and about what>",
  "context": "<1 sentence background to make it feel real>",
  "suggested_elements": ["<element 1>", "<element 2>", "<element 3>", "<element 4>"]
}}"""
    return system, user


# ── Interactive ────────────────────────────────────────────────────────────

def silly_topic_debate(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                       teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.SILLY_TOPIC_DEBATE, age_group)
    user = f"""Generate a silly topic debate lesson.
Topic: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

The student argues FOR the topic. The AI pushes back.
AI lines should provoke extended student responses — not yes/no answers.

Return JSON:
{{
  "topic": "<debate statement>",
  "student_stance": "for",
  "ai_lines": [
    "<AI opening challenge — force student to make first argument>",
    "<AI rebuttal after student's first point — escalate>",
    "<AI final push — demand student's strongest argument>",
    "<AI closing — concede or maintain position>"
  ],
  "debate_hooks": ["<provocative counterargument 1>", "<counterargument 2>"]
}}"""
    return system, user


def roleplay(topic: str, age_group: AgeGroup, difficulty: Difficulty,
             teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.ROLEPLAY, age_group)
    user = f"""Generate a roleplay conversation lesson for spoken English practice.
Scenario: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

AI lines should feel natural, advance the conversation, and require the student to speak substantively.

Return JSON:
{{
  "scenario": "<2-sentence scene description>",
  "student_role": "<student's character>",
  "ai_role": "<AI's character>",
  "ai_lines": [
    "<AI opening line — sets the scene and gives student something to respond to>",
    "<AI follow-up — deepens the conversation>",
    "<AI complication — introduces a small challenge>",
    "<AI closing — wraps up naturally>"
  ],
  "conversation_goal": "<what the student should achieve by the end>"
}}"""
    return system, user


def rapid_fire_qa(topic: str, age_group: AgeGroup, difficulty: Difficulty,
                  teacher_context: TeacherContext) -> tuple[str, str]:
    system = base_system(LessonType.RAPID_FIRE_QA, age_group)
    user = f"""Generate a rapid-fire Q&A lesson for spoken English practice.
Topic: {topic}
Difficulty: {difficulty.value} — {DIFFICULTY_GUIDANCE[difficulty]}{_ctx(teacher_context)}

Questions must be answerable in 1-3 spoken sentences. No trick questions.
AI lines are the prompts/transitions between questions — keep energy high.

Return JSON:
{{
  "topic": "<topic label>",
  "questions": [
    "<question 1>", "<question 2>", "<question 3>",
    "<question 4>", "<question 5>", "<question 6>"
  ],
  "ai_lines": [
    "<AI opening: set the pace and topic>",
    "<AI mid-point: keep energy up>",
    "<AI closing: wrap up with encouragement>"
  ],
  "expected_answer_length": "short"
}}"""
    return system, user


# ── Router ─────────────────────────────────────────────────────────────────

TEMPLATE_MAP = {
    LessonType.PARAGRAPH_READING:  paragraph_reading,
    LessonType.TONGUE_TWISTER:     tongue_twister,
    LessonType.POEM_RECITATION:    poem_recitation,
    LessonType.STORY_BUILDING:     story_building,
    LessonType.YES_AND_IMPROV:     yes_and_improv,
    LessonType.NO_BUT_IMPROV:      no_but_improv,
    LessonType.SHORT_SPEECH:       short_speech,
    LessonType.AUDIO_POSTCARD:     audio_postcard,
    LessonType.SILLY_TOPIC_DEBATE: silly_topic_debate,
    LessonType.ROLEPLAY:           roleplay,
    LessonType.RAPID_FIRE_QA:      rapid_fire_qa,
}


def build_prompt(
    lesson_type: LessonType,
    topic: str,
    age_group: AgeGroup,
    difficulty: Difficulty,
    teacher_context: TeacherContext,
) -> tuple[str, str]:
    fn = TEMPLATE_MAP[lesson_type]
    return fn(topic, age_group, difficulty, teacher_context)