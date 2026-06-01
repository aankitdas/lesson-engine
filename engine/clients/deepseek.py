import json
from openai import OpenAI
from engine.clients.base import GenerationResult
from config.settings import DEEPSEEK_API_KEY, DEEPSEEK_MODEL, cost_usd


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
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.9,
        )
        raw = response.choices[0].message.content or ""
        content, err = _parse_json(raw)
        usage = response.usage
        input_tok  = usage.prompt_tokens
        output_tok = usage.completion_tokens
        return GenerationResult(
            content=content,
            model=DEEPSEEK_MODEL,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cost_usd=cost_usd(DEEPSEEK_MODEL, input_tok, output_tok),
            raw_text=raw,
            error=err,
        )
    except Exception as e:
        return GenerationResult(
            content={}, model=DEEPSEEK_MODEL,
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=str(e),
        )