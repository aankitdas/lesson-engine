import json
from google import genai
from google.genai import types
from engine.clients.base import GenerationResult
from config.settings import GEMINI_API_KEY, GEMINI_MODEL, cost_usd


def _parse_json(text: str) -> tuple[dict, str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text), ""
    except json.JSONDecodeError as e:
        return {}, str(e)


def generate(system: str, user: str) -> GenerationResult:
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.9,
                response_mime_type="application/json",
            ),
        )
        raw = response.text
        content, err = _parse_json(raw)
        usage = response.usage_metadata
        input_tok  = usage.prompt_token_count or 0
        output_tok = usage.candidates_token_count or 0
        return GenerationResult(
            content=content,
            model=GEMINI_MODEL,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost_usd(GEMINI_MODEL, input_tok, output_tok),
            raw_text=raw,
            error=err,
        )
    except Exception as e:
        return GenerationResult(
            content={}, model=GEMINI_MODEL,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=str(e),
        )