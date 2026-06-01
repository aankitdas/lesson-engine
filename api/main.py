from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schema.lesson_types import GenerationRequest, ChalkLesson
from engine.generator import generate_lesson, generate_batch

app = FastAPI(title="Chalk", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=ChalkLesson)
def generate(
    request: GenerationRequest,
    model: str = Query(default="gemini"),
):
    try:
        return generate_lesson(request, model=model)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/batch", response_model=list[ChalkLesson])
def batch(
    request: GenerationRequest,
    model: str = Query(default="gemini"),
):
    if request.batch_size < 1 or request.batch_size > 10:
        raise HTTPException(status_code=400, detail="batch_size must be 1–10")
    results = generate_batch(request, model=model)
    if not results:
        raise HTTPException(status_code=500, detail="All generations failed")
    return results


@app.get("/lesson-types")
def lesson_types():
    from schema.lesson_types import LessonType, AgeGroup, Difficulty
    return {
        "lesson_types": [t.value for t in LessonType],
        "age_groups":   [a.value for a in AgeGroup],
        "difficulties": [d.value for d in Difficulty],
    }