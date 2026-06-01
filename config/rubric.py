# Rubric constants — single source of truth for all of Chalk

# ── Metric weights per module ──────────────────────────────────────────────
MODULE_WEIGHTS = {
    "paragraph_reading":   {"P": 0.24, "Pr": 0.24, "R": 0.14, "Ff": 0.14, "V": 0.19, "TA": 0.05},
    "tongue_twister":      {"P": 0.28, "Pr": 0.14, "R": 0.24, "Ff": 0.10, "V": 0.19, "TA": 0.05},
    "poem_recitation":     {"P": 0.19, "Pr": 0.29, "R": 0.14, "Ff": 0.14, "V": 0.19, "TA": 0.05},
    "roleplay":            {"P": 0.20, "Pr": 0.16, "R": 0.12, "Ff": 0.16, "V": 0.16, "TA": 0.20},
    "silly_topic_debate":  {"P": 0.16, "Pr": 0.20, "R": 0.16, "Ff": 0.12, "V": 0.16, "TA": 0.20},
    "rapid_fire_qa":       {"P": 0.12, "Pr": 0.12, "R": 0.20, "Ff": 0.20, "V": 0.16, "TA": 0.20},
    "yes_and_improv":      {"P": 0.16, "Pr": 0.20, "R": 0.12, "Ff": 0.16, "V": 0.16, "TA": 0.20},
    "no_but_improv":       {"P": 0.12, "Pr": 0.20, "R": 0.16, "Ff": 0.16, "V": 0.16, "TA": 0.20},
    "story_building":      {"P": 0.16, "Pr": 0.16, "R": 0.16, "Ff": 0.16, "V": 0.16, "TA": 0.20},
    "short_speech":        {"P": 0.16, "Pr": 0.16, "R": 0.16, "Ff": 0.12, "V": 0.20, "TA": 0.20},
    "audio_postcard":      {"P": 0.16, "Pr": 0.16, "R": 0.12, "Ff": 0.20, "V": 0.16, "TA": 0.20},
}

# Dominant metric per module (used to steer generation)
def _top_metrics(weights: dict) -> list[str]:
    top_val = max(weights.values())
    return [m for m, w in weights.items() if w == top_val]

DOMINANT_METRICS = {k: _top_metrics(v) for k, v in MODULE_WEIGHTS.items()}
# ── Score bands ────────────────────────────────────────────────────────────
SCORE_BANDS = [
    (90, 100, "Excellent"),
    (75,  89, "Strong"),
    (50,  74, "Adequate"),
    (0,   49, "Needs Improvement"),
]

# Per-marker bands (pronunciation uses different cutoffs)
MARKER_BANDS = {
    "P":  [(82, 100, "Excellent"), (67, 81, "Strong"), (47, 66, "Adequate"), (0, 46, "Poor")],
    "default": [(90, 100, "Excellent"), (75, 89, "Strong"), (50, 74, "Adequate"), (0, 49, "Poor")],
}

# ── Age grading modes ──────────────────────────────────────────────────────
AGE_GROUPS = {
    "child":  {"label": "≤12",  "mode": "LENIENT",       "rule": "Focus on effort; ignore minor errors; don't penalize slow pace or fillers"},
    "teen":   {"label": "13-15","mode": "BALANCED",      "rule": "Allow minor fluency issues; content clarity over polish"},
    "youth":  {"label": "16-18","mode": "STANDARD",      "rule": "Expect clear, confident, expressive speech"},
    "adult":  {"label": "19+",  "mode": "PROFESSIONAL",  "rule": "Penalize poor pacing, weak articulation, lack of structure"},
}

# ── Gender pitch anchors ───────────────────────────────────────────────────
PITCH_ANCHORS = {
    "male":    {"range": (80, 200),  "optimal": 140},
    "female":  {"range": (150, 300), "optimal": 225},
    "neutral": {"range": (100, 260), "optimal": 180},
}

# ── Error patterns (targeted generation) ──────────────────────────────────
ERROR_PATTERNS = {
    "s_sh_confusion":          {"description": "consonant substitution",         "severity": "moderate"},
    "th_t_substitution":       {"description": "replacing /θ/ with /t/",         "severity": "high"},
    "consonant_cluster_reduction": {"description": "simplifying blends (str→s)", "severity": "high"},
}

# ── Scoring rules ──────────────────────────────────────────────────────────
STRICT_CAP = 85               # score = min(score, 85) unless genuinely excellent
INFLATION_THRESHOLD = 85      # rolling mean over 200 evals
INFLATION_MULTIPLIER = 0.9
TONGUE_TWISTER_MIN_REPS = 3   # completion = 0 if fewer

# ── Quality gate (Chalk internal) ─────────────────────────────────────────
CHALK_PASS_THRESHOLD = 0.75
CHALK_MAX_RETRIES = 3
CHALK_DEDUP_THRESHOLD = 0.85  # cosine similarity ceiling