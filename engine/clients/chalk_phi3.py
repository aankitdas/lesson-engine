import json
from openai import OpenAI
from engine.clients.base import GenerationResult
import re

_BASE_URL = "http://localhost:8001/v1"

def _parse_json(text: str) -> tuple[dict, str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # try full parse
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass

    # try extracting complete JSON object
    start = text.find("{")
    if start == -1:
        return {}, "No JSON found"

    # try progressively truncating to find valid JSON
    end = len(text)
    while end > start:
        candidate = text[start:end]
        # try as-is
        try:
            return json.loads(candidate), ""
        except json.JSONDecodeError:
            pass
        # try closing with }
        try:
            return json.loads(candidate + "}"), ""
        except json.JSONDecodeError:
            pass
        # step back to last comma
        last_comma = candidate.rfind(",")
        if last_comma == -1:
            break
        end = start + last_comma

    return {}, "Could not parse JSON"

import re

def _extract_fields(text: str) -> dict:
    result = {}
    for m in re.finditer(r'"(\w+)":\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2).replace('\\"', '"')
    for m in re.finditer(r'"(\w+)":\s*(\d+)', text):
        result[m.group(1)] = int(m.group(2))
    return result

def _parse_json(text: str) -> tuple[dict, str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text), ""
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end]), ""
        except json.JSONDecodeError:
            pass
    # last resort: extract individual fields
    fields = _extract_fields(text)
    if fields:
        return fields, ""
    return {}, "No JSON found"
    
def generate(system: str, user: str) -> GenerationResult:
    client = OpenAI(base_url=_BASE_URL, api_key="chalk")
    try:
        response = client.chat.completions.create(
            model="chalk-phi3",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        content, err = _parse_json(raw)
        usage = response.usage
        return GenerationResult(
            content=content,
            model="chalk-phi3-ft",
            input_tokens=usage.prompt_tokens     if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cost_usd=0.0,
            raw_text=raw,
            error=err,
        )
    except Exception as e:
        return GenerationResult(
            content={}, model="chalk-phi3-ft",
            input_tokens=0, output_tokens=0, cost_usd=0.0,
            error=str(e),
        )