import json
from openai import OpenAI
from engine.clients.base import GenerationResult

_BASE_URL = "http://localhost:11434/v1"


def _parse_json(text: str) -> tuple[dict, str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        # local models sometimes wrap JSON in prose — try to extract it
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end]), ""
            except json.JSONDecodeError as e:
                return {}, str(e)
        return {}, "No JSON object found in response"


def generate(system: str, user: str, model: str = "llama3.1:8b") -> GenerationResult:
    client = OpenAI(base_url=_BASE_URL, api_key="ollama")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.9,
        )
        raw = response.choices[0].message.content or ""
        content, err = _parse_json(raw)
        usage = response.usage
        return GenerationResult(
            content=content,
            model=f"ollama/{model}",
            input_tokens=usage.prompt_tokens     if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=0.0,  # local — no API cost
            raw_text=raw,
            error=err,
        )
    except Exception as e:
        return GenerationResult(
            content={}, model=f"ollama/{model}",
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=str(e),
        )