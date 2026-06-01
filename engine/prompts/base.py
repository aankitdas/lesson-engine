from config.rubric import MODULE_WEIGHTS, DOMINANT_METRICS, AGE_GROUPS
from schema.lesson_types import AgeGroup, Difficulty, LessonType, TeacherContext

# Short, token-efficient metric guidance injected per lesson type
METRIC_GUIDANCE = {
    "P":  "pronunciation — include consonant clusters, target phonemes, syllable-rich words",
    "Pr": "prosody — vary sentence length, include questions/exclamations, mark natural stress points",
    "R":  "speaking rate — mix short punchy lines with longer ones, add natural pause cues",
    "Ff": "fluency — use clear, unambiguous prompts that reduce hesitation and repair",
    "V":  "volume — include moments of emphasis, emotional peaks, and projection cues",
    "TA": "task adherence — make constraints crystal clear, reward creativity within the task",
}

AGE_TONE = {
    AgeGroup.CHILD:  "Use simple vocabulary (Grade 1-5), short sentences, fun and playful tone.",
    AgeGroup.TEEN:   "Use everyday language (Grade 6-9), relatable scenarios, light humour OK.",
    AgeGroup.YOUTH:  "Use natural conversational language (Grade 10-12), nuanced topics welcome.",
    AgeGroup.ADULT:  "Use sophisticated vocabulary, complex sentence structures, professional contexts.",
}

DIFFICULTY_GUIDANCE = {
    Difficulty.EASY:   "Keep phoneme targets simple, sentence length short, topic familiar.",
    Difficulty.MEDIUM: "Introduce moderate phoneme complexity, varied sentence rhythm, some unfamiliar vocabulary.",
    Difficulty.HARD:   "Dense phoneme clusters, long complex sentences, challenging vocabulary and concepts.",
}


def build_metric_guidance(lesson_type: LessonType) -> str:
    dominant = DOMINANT_METRICS.get(lesson_type.value, [])
    weights = MODULE_WEIGHTS.get(lesson_type.value, {})
    lines = []
    for metric in dominant:
        lines.append(f"  - {metric}: {METRIC_GUIDANCE[metric]}")
    # also include secondary metrics above 0.15
    for metric, weight in sorted(weights.items(), key=lambda x: -x[1]):
        if metric not in dominant and weight >= 0.15:
            lines.append(f"  - {metric} (secondary, {weight}): {METRIC_GUIDANCE[metric]}")
    return "\n".join(lines)


def build_teacher_context(ctx: TeacherContext) -> str:
    parts = []
    if ctx.buzzwords:
        parts.append(f"Weave in these buzzwords/references naturally: {', '.join(ctx.buzzwords)}")
    if ctx.event_note:
        parts.append(f"Tie to this event or occasion: {ctx.event_note}")
    if ctx.book_excerpt:
        parts.append(f"Draw from this text:\n---\n{ctx.book_excerpt[:800]}\n---")
    if ctx.extra_notes:
        parts.append(f"Teacher notes: {ctx.extra_notes}")
    return "\n".join(parts) if parts else ""


SPEAKING_MODE = {
    "paragraph_reading":  "Students will READ this passage aloud — every word choice affects vocal delivery.",
    "tongue_twister":     "Students will SPEAK this rapidly aloud — phoneme density and rhythm are critical.",
    "poem_recitation":    "Students will RECITE this poem aloud — prosody and stress patterns are critical.",
    "short_speech":       "Students will DELIVER this speech aloud — structure and vocal presence matter.",
    "audio_postcard":     "Students will NARRATE this aloud — natural flow and personal tone matter.",
    "story_building":     "Students will IMPROVISE spoken story continuations — prompts must spark spontaneous speech.",
    "yes_and_improv":     "Students will IMPROVISE spoken responses — prompts must spark spontaneous, extended speech.",
    "no_but_improv":      "Students will IMPROVISE spoken rebuttals — prompts must demand reasoning and extended response.",
    "silly_topic_debate": "Students will ARGUE their position aloud — content must provoke extended spoken responses.",
    "roleplay":           "Students will SPEAK in character — scenarios must feel natural and demand real conversation.",
    "rapid_fire_qa":      "Students will ANSWER rapidly aloud — questions must demand quick, clear spoken responses.",
}

def base_system(lesson_type: LessonType, age_group: AgeGroup) -> str:
    speaking_line = SPEAKING_MODE.get(lesson_type.value, "Students will SPEAK this content aloud.")
    return f"""You are a specialist K-12 spoken English lesson designer.
{speaking_line}

Age group: {AGE_GROUPS[age_group.value]['label']} — {AGE_GROUPS[age_group.value]['mode']}
{AGE_TONE[age_group]}

Optimise primarily for these voice metrics:
{build_metric_guidance(lesson_type)}

Respond ONLY with valid JSON. No markdown, no explanation, no extra keys."""